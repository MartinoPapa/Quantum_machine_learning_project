import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import networkx as nx
import pandas as pd
import os
import sys
import pickle
from sklearn.model_selection import train_test_split

# ==========================================
# 0. DUAL LOGGING SETUP
# ==========================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("output.txt")

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================

# --- Dataset Parameters ---
TRAINING_DIM = 700                 # Number of graphs for training
VALIDATION_DIM = 10                 # Number of graphs for validation
TEST_DIM = 100                      # Number of graphs for testing
FRAUD_RATIO = 0.3                  # Target percentage of Fraud Graphs (0.3 = 30%)
SAMPLED_EDGES = 8                 # Minimum edges in sampled subgraphs

# --- Graph Generation Parameters ---
TOTAL_GRAPHS = TRAINING_DIM + VALIDATION_DIM + TEST_DIM
NUM_NODES = 16                     
NUM_QUBITS = int(np.log2(NUM_NODES)) 
TOTAL_QUBITS = 2 * NUM_QUBITS      

# --- Training/Test Split ---
BATCH_SIZE = 16                    

# --- Model Architecture ---
NUM_LAYERS = 2                     
N_BETTI = 1                        
LATENT_DIM = 2 + (N_BETTI + 1) * 3 

# --- Memory Bank Parameters ---
K_MEMORY_BANK = 10                 
BANK_PERCENTAGE_UPDATE = 0.1       
WARMUP_STEPS = 5                   

# --- Optimization Hyperparameters ---
EPOCHS = 20                      
LEARNING_RATE = 0.005           

# --- Loss Function Weights ---
LAMBDA_1 = 0.1                     
LAMBDA_2 = 0.01                   

# --- Circuit Initialization ---
THETA_0_INIT = 1.0                 
THETA_1_INIT = -5.0                

dev = qml.device("default.mixed", wires=TOTAL_QUBITS)

# ==========================================
# PRINT CONFIGURATION
# ==========================================
print("="*60)
print("              QTGNN DIRECTED CONFIGURATION")
print("="*60)
print(f"[DATASET SPLIT]")
print(f"  Training Graphs:       {TRAINING_DIM}")
print(f"  Validation Graphs:     {VALIDATION_DIM}")
print(f"  Test Graphs:           {TEST_DIM}")
print(f"  Total Graphs:          {TOTAL_GRAPHS}")
print(f"  Target Fraud Ratio:    {FRAUD_RATIO*100}%")
print("-" * 60)
print(f"[GRAPH CONSTRUCTION]")
print(f"  Sampled Min Edges:     {SAMPLED_EDGES}")
print(f"  Nodes per Subgraph:    {NUM_NODES}")
print(f"  Sampling Strategy:     BFS (Snowball) from Hubs")
print("-" * 60)
print(f"[MODEL ARCHITECTURE]")
print(f"  VQGC Layers:           {NUM_LAYERS}")
print(f"  Betti Thresholds:      {N_BETTI}")
print(f"  Latent Dimension:      {LATENT_DIM}")
print("="*60 + "\n")

# ==========================================
# 2. DATASET (FRAUD-BALANCED BFS SAMPLING)
# ==========================================

class PaySimDataset:
    def __init__(self, csv_file, pool_file="pool_nodes.pkl"):
        self.graphs = []
        self.labels = []
        self.num_nodes = NUM_NODES
        self.G_global = nx.DiGraph() 
        
        # 1. Load the Filter List
        if os.path.exists(pool_file):
            print(f"Loading Node Pool from {pool_file}...")
            with open(pool_file, "rb") as f:
                self.allowed_nodes = pickle.load(f)
            print(f"Filter active: Only loading edges connecting these {len(self.allowed_nodes)} nodes.")
        else:
            raise FileNotFoundError(f"Please run 'find_pool_of_components.py' first to generate {pool_file}")

        # 2. Load and Filter Data
        print(f"Loading and Filtering {csv_file}...")
        self._build_filtered_graph(csv_file)
        
        print(f"Global Pool Graph: {self.G_global.number_of_nodes()} nodes, {self.G_global.number_of_edges()} edges.")
        
        # 3. Sample Subgraphs
        print(f"Sampling {TOTAL_GRAPHS} subgraphs (Target Fraud: {int(TOTAL_GRAPHS * FRAUD_RATIO)})...")
        self._sample_subgraphs_bfs_balanced()

    def _build_filtered_graph(self, csv_file):
        """
        Loads the CSV but discards any edge that isn't in our allowed_nodes set.
        """
        chunk_size = 500000
        chunk_iter = pd.read_csv(csv_file, chunksize=chunk_size, 
                                 usecols=['type', 'amount', 'nameOrig', 'nameDest', 'isFraud'])
        
        for i, chunk in enumerate(chunk_iter):
            mask = chunk['nameOrig'].isin(self.allowed_nodes) & chunk['nameDest'].isin(self.allowed_nodes)
            filtered_chunk = chunk[mask]
            
            if filtered_chunk.empty: continue

            filtered_chunk = filtered_chunk.copy()
            filtered_chunk['amount'] = np.log1p(filtered_chunk['amount'])
            
            for row in filtered_chunk.itertuples():
                u, v = row.nameOrig, row.nameDest
                w, f = row.amount, row.isFraud
                
                if self.G_global.has_edge(u, v):
                    self.G_global[u][v]['weight'] += w
                    # If any transaction is fraud, the edge is fraud
                    if f == 1: self.G_global[u][v]['isFraud'] = 1
                else:
                    self.G_global.add_edge(u, v, weight=w, isFraud=f)
            
            if (i+1) % 5 == 0:
                print(f"Scanned chunk {i+1}... Graph size: {self.G_global.number_of_edges()} edges")

    def _sample_subgraphs_bfs_balanced(self):
        """
        Uses BFS (Snowball Sampling) to maximize density.
        Explicitly targets fraud nodes first to meet the FRAUD_RATIO.
        """
        # 1. Identify Fraud Nodes and Normal Hubs
        # Fraud nodes = any node involved in a fraud edge
        fraud_edges = [ (u,v) for u,v,d in self.G_global.edges(data=True) if d.get('isFraud', 0) == 1 ]
        fraud_starts = list(set([u for u,v in fraud_edges] + [v for u,v in fraud_edges]))
        
        # Normal starts = everyone else, sorted by degree (Hubs first)
        all_nodes = list(self.G_global.nodes())
        degrees = dict(self.G_global.degree())
        normal_starts = [n for n in all_nodes if n not in fraud_starts]
        normal_starts.sort(key=lambda n: degrees[n], reverse=True) # Prioritize Hubs
        
        # Randomize fraud starts slightly to avoid deterministic order
        np.random.shuffle(fraud_starts)
        
        target_fraud_count = int(TOTAL_GRAPHS * FRAUD_RATIO)
        target_normal_count = TOTAL_GRAPHS - target_fraud_count
        
        print(f"Sampling Plan -> Fraud: {target_fraud_count} | Normal: {target_normal_count}")
        
        self.graphs = []
        self.labels = []
        
        # --- PHASE 1: COLLECT FRAUD GRAPHS ---
        count_fraud = 0
        G_undirected = self.G_global.to_undirected(as_view=True)
        
        # We cycle through fraud_starts if needed
        fraud_idx = 0
        while count_fraud < target_fraud_count and fraud_idx < len(fraud_starts) * 5:
            # Wrap around if we run out of unique starts
            start_node = fraud_starts[fraud_idx % len(fraud_starts)]
            fraud_idx += 1
            
            subG = self._bfs_expansion(start_node, G_undirected, degrees)
            
            if self._is_valid_subgraph(subG):
                # Check label
                f_flags = nx.get_edge_attributes(subG, 'isFraud').values()
                if 1 in f_flags:
                    self.graphs.append(subG)
                    self.labels.append(1)
                    count_fraud += 1
                    if count_fraud % 10 == 0: print(f"  [Fraud] Collected {count_fraud}/{target_fraud_count}")
        
        print(f"Phase 1 Complete. Collected {count_fraud} Fraud Graphs.")
        
        # --- PHASE 2: COLLECT NORMAL GRAPHS ---
        count_normal = 0
        norm_idx = 0
        
        while count_normal < target_normal_count and norm_idx < len(normal_starts):
            start_node = normal_starts[norm_idx]
            norm_idx += 1
            
            subG = self._bfs_expansion(start_node, G_undirected, degrees)
            
            if self._is_valid_subgraph(subG):
                f_flags = nx.get_edge_attributes(subG, 'isFraud').values()
                # Only accept if truly normal (label 0)
                if 1 not in f_flags:
                    self.graphs.append(subG)
                    self.labels.append(0)
                    count_normal += 1
                    if count_normal % 50 == 0: print(f"  [Normal] Collected {count_normal}/{target_normal_count}")

        print(f"Phase 2 Complete. Collected {count_normal} Normal Graphs.")
        print(f"Total Graphs Generated: {len(self.graphs)} (Fraud: {sum(self.labels)})")

    def _bfs_expansion(self, start_node, G_view, degrees):
        """Helper: Performs BFS expansion from start_node to find dense cluster"""
        sampled_nodes = {start_node}
        queue = [start_node]
        
        while len(sampled_nodes) < self.num_nodes and queue:
            curr = queue.pop(0)
            neighbors = list(G_view.neighbors(curr))
            # Sort neighbors by degree to prefer hubs
            neighbors.sort(key=lambda n: degrees[n], reverse=True)
            
            for n in neighbors:
                if n not in sampled_nodes:
                    sampled_nodes.add(n)
                    queue.append(n)
                    if len(sampled_nodes) >= self.num_nodes: break
        
        return self.G_global.subgraph(list(sampled_nodes)).copy()

    def _is_valid_subgraph(self, subG):
        """Helper: Checks edge count and connectivity"""
        if subG.number_of_nodes() < 4: return False
        if subG.number_of_edges() < SAMPLED_EDGES: return False
        
        if not nx.is_weakly_connected(subG):
            # Try to keep largest component
            largest_cc = max(nx.weakly_connected_components(subG), key=len)
            if len(largest_cc) < self.num_nodes * 0.8: return False
            # Note: We don't modify subG here to keep logic simple, just reject disconnected
            # or allow slightly disconnected if dense enough?
            # Let's enforce strict weak connectivity for TDA
            return False
            
        return True

    def get_matrix_A(self, g):
        g = nx.convert_node_labels_to_integers(g)
        N = self.num_nodes
        adj = np.zeros((N, N)) 
        
        if g.number_of_nodes() > 0:
            nodes = sorted(list(g.nodes))
            for u, v, data in g.edges(data=True):
                i, j = nodes.index(u), nodes.index(v)
                w = data.get('weight', 0)
                adj[i, j] = w
            
            in_deg = dict(g.in_degree(weight='weight'))
            out_deg = dict(g.out_degree(weight='weight'))
            total_weight_sum = g.size(weight='weight') + 1e-9
            
            for i, node in enumerate(nodes):
                deg = in_deg.get(node, 0) + out_deg.get(node, 0)
                adj[i, i] = deg / total_weight_sum

        feature_vec = adj.flatten() 
        target_dim = 2**TOTAL_QUBITS 
        if len(feature_vec) < target_dim:
            feature_vec = np.pad(feature_vec, (0, target_dim - len(feature_vec)))
        else:
            feature_vec = feature_vec[:target_dim]
        norm = np.linalg.norm(feature_vec)
        if norm < 1e-9: norm = 1.0
        return feature_vec / norm

# ==========================================
# 3. MEMORY BANK
# ==========================================

class MemoryBank:
    def __init__(self, capacity, feature_dim):
        self.capacity = capacity
        self.bank = torch.empty(0, feature_dim) 
        
    def update(self, new_features):
        if new_features.shape[0] == 0: return
        new_entries = new_features.detach().cpu()
        max_update = max(1, int(self.capacity * BANK_PERCENTAGE_UPDATE))
        if new_entries.shape[0] > max_update:
            indices = torch.randperm(new_entries.shape[0])[:max_update]
            entries_to_add = new_entries[indices]
        else:
            entries_to_add = new_entries
        combined = torch.cat([self.bank, entries_to_add], dim=0)
        if combined.shape[0] > self.capacity:
            self.bank = combined[-self.capacity:]
        else:
            self.bank = combined

    def get_nearest_distance(self, features):
        if self.bank.shape[0] == 0:
            return torch.zeros(features.shape[0], device=features.device)
        bank_device = self.bank.to(features.device)
        dists = torch.cdist(features, bank_device, p=2).pow(2)
        min_dists, _ = torch.min(dists, dim=1)
        return min_dists

# ==========================================
# 4. QUANTUM CIRCUIT
# ==========================================

@qml.qnode(dev, interface="torch")
def qtgnn_circuit(inputs, theta_e, theta_h, theta_p, layers):
    qml.AmplitudeEmbedding(features=inputs, wires=range(TOTAL_QUBITS), normalize=True)
    for i in range(NUM_QUBITS):
        qml.IsingXX(theta_e[i], wires=[i, i + NUM_QUBITS])
    sys_wires = range(NUM_QUBITS)
    for l in range(layers):
        for i in sys_wires:
            qml.RZ(theta_h[l, 0, i], wires=i)
        cnt = 0
        for i in range(NUM_QUBITS):
            for j in range(i + 1, NUM_QUBITS):
                qml.IsingZZ(theta_h[l, 1, cnt], wires=[i, j])
                cnt = (cnt + 1) % theta_h.shape[-1]
        t0 = theta_p[l, 0]
        t1 = theta_p[l, 1]
        denom = torch.exp(t0) + torch.exp(t1)
        p1 = torch.exp(t1) / denom 
        for i in sys_wires:
            qml.PhaseDamping(p1, wires=i)
    return qml.density_matrix(wires=sys_wires)

# ==========================================
# 5. TOPOLOGICAL FEATURES
# ==========================================

def compute_topological_features(rho_np, epsilon_steps=3):
    N = rho_np.shape[0]
    dists = np.zeros((N, N))
    rho_abs = np.abs(rho_np)
    for i in range(N):
        for j in range(i+1, N):
            denom = np.sqrt(rho_abs[i,i] * rho_abs[j,j]) + 1e-9
            fid = rho_abs[i,j] / denom
            d = np.sqrt(max(0, 1 - fid**2))
            dists[i,j] = dists[j,i] = d
    features = []
    epsilons = np.linspace(0.1, 0.9, epsilon_steps)
    for eps in epsilons:
        adj = (dists <= eps).astype(int)
        np.fill_diagonal(adj, 0)
        G = nx.from_numpy_array(adj)
        b0 = nx.number_connected_components(G)
        b1 = len(nx.cycle_basis(G))
        b2 = sum(1 for _ in nx.enumerate_all_cliques(G) if len(_) == 3)
        chi = b0 - b1 + b2
        features.extend([b0/N, b1/N, b2/N, chi/N])
    return torch.tensor(features, dtype=torch.float32)

# ==========================================
# 6. HYBRID MODEL
# ==========================================

class QTGNN(nn.Module):
    def __init__(self, num_nodes, num_layers=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_qubits = NUM_QUBITS
        self.layers = num_layers
        self.theta_e = nn.Parameter(torch.randn(self.n_qubits) * 0.1)
        max_params = self.n_qubits * (self.n_qubits - 1) // 2
        self.theta_h = nn.Parameter(torch.randn(num_layers, 2, max_params) * 0.1)
        self.theta_p = nn.Parameter(torch.tensor([[THETA_0_INIT, THETA_1_INIT]] * num_layers))
        self.feat_dim = 2 + 4 * 3 
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 1)
        )
        self.memory_bank = MemoryBank(K_MEMORY_BANK, self.feat_dim)

    def get_quantum_features(self, rho):
        rho = (rho + rho.conj().transpose(-1, -2)) / 2.0
        z_q = torch.real(torch.trace(rho @ rho))
        try:
            _, evals, _ = torch.linalg.svd(rho)
        except:
            rho_jitter = rho + 1e-7 * torch.eye(rho.shape[-1], device=rho.device)
            _, evals, _ = torch.linalg.svd(rho_jitter)
        evals = torch.clamp(evals, min=1e-9)
        evals = evals / torch.sum(evals)
        c_q = -torch.sum(evals * torch.log(evals))
        return torch.stack([z_q, c_q]).float()

    def forward(self, inputs):
        rho = qtgnn_circuit(inputs, self.theta_e, self.theta_h, self.theta_p, self.layers)
        q_feats = self.get_quantum_features(rho)
        with torch.no_grad():
            rho_np = rho.detach().cpu().numpy()
            tda_feats = compute_topological_features(rho_np).to(inputs.device)
        phi = torch.cat([q_feats, tda_feats])
        logits = self.classifier(phi)
        return logits, torch.sigmoid(logits), phi

# ==========================================
# 7. EXECUTION
# ==========================================

def run_experiment():
    CSV_FILE = "PS_20174392719_1491204439457_log.csv"
    try:
        dataset = PaySimDataset(CSV_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    graphs = dataset.graphs
    labels = dataset.labels
    
    total_avail = len(graphs)
    if total_avail == 0:
        print("Error: No graphs generated.")
        return

    # Indices
    indices = np.arange(total_avail)
    
    # Stratified Split to maintain FRAUD_RATIO across sets if possible
    # But usually we just shuffle. Given the targeted sampling, shuffle is fine.
    np.random.shuffle(indices)
    
    end_train = min(total_avail, TRAINING_DIM)
    end_val = min(total_avail, TRAINING_DIM + VALIDATION_DIM)
    
    train_indices = indices[:end_train]
    val_indices = indices[end_train:end_val]
    test_indices = indices[end_val:]
    
    print(f"Split -> Train: {len(train_indices)} | Val (S_normal): {len(val_indices)} | Test: {len(test_indices)}")
    
    # Filter Val indices (Keep only Normal for S_normal)
    val_indices_normal = [i for i in val_indices if labels[i] == 0]
    if len(val_indices_normal) < 2:
        print("Warning: Validation set has too few normal graphs. Threshold might be unstable.")

    model = QTGNN(num_nodes=NUM_NODES, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_labels = [labels[i] for i in train_indices]
    n_neg = train_labels.count(0)
    n_pos = train_labels.count(1)
    
    # Safe weight calculation
    if n_pos == 0:
        pos_weight = torch.tensor([1.0])
    else:
        # Heavily weight the minority class
        pos_weight = torch.tensor([n_neg / n_pos])
        
    print(f"Training Stats: Normal={n_neg}, Fraud={n_pos}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print("\n--- Starting Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_normal_phis = [] 
        np.random.shuffle(train_indices)
        
        for i in train_indices:
            inputs = torch.tensor(dataset.get_matrix_A(graphs[i]), dtype=torch.float32)
            label = torch.tensor([labels[i]], dtype=torch.float32)
            
            optimizer.zero_grad()
            logits, probs, phi = model(inputs)
            
            l_sup = criterion(logits, label)
            
            if label.item() == 0:
                dist_sq = model.memory_bank.get_nearest_distance(phi.unsqueeze(0))
                l_unsup = dist_sq.mean()
                batch_normal_phis.append(phi)
            else:
                l_unsup = torch.tensor(0.0)
            
            l_reg = sum(torch.norm(p)**2 for p in model.parameters())
            loss = l_sup + (LAMBDA_1 * l_unsup) + (LAMBDA_2 * l_reg)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if len(batch_normal_phis) > 0:
            with torch.no_grad():
                new_entries = torch.stack(batch_normal_phis)
                model.memory_bank.update(new_entries)
        
        avg_loss = total_loss/max(1, len(train_indices))
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    print("\n--- Populating S_normal and Calculating Threshold ---")
    model.eval()
    S_normal_list = []
    
    with torch.no_grad():
        for i in val_indices_normal:
            inputs = torch.tensor(dataset.get_matrix_A(graphs[i]), dtype=torch.float32)
            _, _, phi = model(inputs)
            S_normal_list.append(phi)
            
    if not S_normal_list:
        print("Error: No normal graphs in validation set.")
        return

    S_normal_tensor = torch.stack(S_normal_list)
    TAU_THRESHOLD = 0.5
    print(f"Dynamic Threshold (Tau): {TAU_THRESHOLD:.6f}")

    print(f"\n--- Testing ---")
    correct = 0
    tp = 0; fp = 0; tn = 0; fn = 0
    scores_log = []
    
    if len(test_indices) == 0:
        print("No test indices available.")
        return

    with torch.no_grad():
        for i in test_indices:
            inputs = torch.tensor(dataset.get_matrix_A(graphs[i]), dtype=torch.float32)
            label = labels[i]
            _, _, phi = model(inputs)
            
            dists = torch.norm(S_normal_tensor - phi, dim=1).pow(2)
            s_G = torch.min(dists).item()
            scores_log.append(s_G)
            
            pred_class = 1 if s_G > TAU_THRESHOLD else 0
            
            if pred_class == label: correct += 1
            if label == 1 and pred_class == 1: tp += 1
            if label == 0 and pred_class == 1: fp += 1
            if label == 0 and pred_class == 0: tn += 1
            if label == 1 and pred_class == 0: fn += 1
            
            lbl_str = "FRAUD " if label == 1 else "NORMAL"
            print(f"Graph {i:03d} | {lbl_str} | Score: {s_G:.6f} | Pred: {pred_class}")
            print(f"Phi({i:03d}) = {phi}")

    acc = 100*correct/len(test_indices)
    print(f"\nFinal Accuracy: {acc:.1f}%")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

if __name__ == "__main__":
    run_experiment()