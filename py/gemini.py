import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import networkx as nx
import pandas as pd
import os
import sys
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
# 1. CONFIGURATION
# ==========================================

# --- Data & Graph Parameters ---
MAX_ROWS = 500000                  
TOTAL_GRAPHS = 300 
PERCENTAGE_FRAUD = 0.1
BANK_PERCENTAGE_UPDATE = 0.1
TEST_SIZE = 0.2       
NUM_NODES = 16                     
NUM_QUBITS = int(np.log2(NUM_NODES)) 
TOTAL_QUBITS = 2 * NUM_QUBITS      
BATCH_SIZE = 16

# --- Model Architecture ---
NUM_LAYERS = 3 #Number of layers in the Variational Quantum Graph Convolution                   
N_BETTI = 2                         
LATENT_DIM = 2 + (N_BETTI + 1) * 3    

# --- Memory Bank Parameters ---
K_MEMORY_BANK = 10                 # Larger bank to hold the "Normal" manifold
WARMUP_STEPS = 5                      

# --- Validation/Inference Parameters ---
M_VALIDATION_S_NORMAL = 10         

# --- Optimization Hyperparameters ---
EPOCHS = 20                        
LEARNING_RATE = 0.005              # Lower LR for stability

# FIX 1: Reduced Unsupervised Weight
LAMBDA_1 = 0.1                     
LAMBDA_2 = 0.001                      

THETA_0_INIT = 1.0    
THETA_1_INIT = -5.0                   

dev = qml.device("default.mixed", wires=TOTAL_QUBITS)

print(f"--- QTGNN Final Implementation ---")
print(f"Total Qubits: {TOTAL_QUBITS}")
print(f"Nodes per Graph: {NUM_NODES}")
print(f"Lambda_1 (Unsup): {LAMBDA_1}")
print(f"Lambda_2 (regularization): {LAMBDA_2}")
print(f"Output: Console & output.txt")
print(f"-------------------------------------------")

# ==========================================
# 2. DATASET
# ==========================================

class PaySimDataset:
    def __init__(self, csv_file, total_graphs=TOTAL_GRAPHS, max_rows=MAX_ROWS):
        self.graphs = []
        self.labels = []
        self.num_nodes = NUM_NODES
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"File {csv_file} not found.")

        print(f"Loading {csv_file}...")
        df = pd.read_csv(csv_file, nrows=max_rows, 
                         usecols=['type', 'amount', 'nameOrig', 'nameDest', 'isFraud'])
        
        df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].reset_index(drop=True)
        
        # Log-Normalize Amount
        df['amount'] = np.log1p(df['amount'])
        df['amount'] = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min() + 1e-9)
        
        fraud_indices = df[df['isFraud'] == 1].index.tolist()
        normal_indices = df[df['isFraud'] == 0].index.tolist()
        
        # FIX: 80% Normal / 20% Fraud split
        target_fraud = int(total_graphs * PERCENTAGE_FRAUD)
        target_normal = total_graphs - target_fraud
        
        print(f"Generating {target_fraud} Fraud and {target_normal} Normal graphs...")
        
        self._generate_graphs(df, fraud_indices, target_fraud, is_fraud_target=True)
        self._generate_graphs(df, normal_indices, target_normal, is_fraud_target=False)

    def _generate_graphs(self, df, available_indices, target_count, is_fraud_target):
        count = 0
        np.random.shuffle(available_indices)
        
        for start_idx in available_indices:
            if count >= target_count: break
            
            G = nx.Graph()
            visited_tx = {start_idx}
            
            row = df.loc[start_idx]
            G.add_edge(row['nameOrig'], row['nameDest'], weight=row['amount'], isFraud=row['isFraud'])
            
            attempts = 0
            MAX_ATTEMPTS = self.num_nodes * 3 
            
            while G.number_of_nodes() < self.num_nodes and attempts < MAX_ATTEMPTS:
                attempts += 1
                current_nodes = list(G.nodes)
                
                candidates = df[
                    (df['nameOrig'].isin(current_nodes)) | 
                    (df['nameDest'].isin(current_nodes))
                ]
                
                candidates = candidates[~candidates.index.isin(visited_tx)]
                
                if candidates.empty:
                    break 
                
                # --- Connectivity Score (Triadic Closure) ---
                def connectivity_score(row):
                    matches = 0
                    if row['nameOrig'] in current_nodes: matches += 1
                    if row['nameDest'] in current_nodes: matches += 1
                    return matches

                if len(candidates) > 50:
                    candidates_subset = candidates.sample(n=50)
                else:
                    candidates_subset = candidates.copy()
                
                scores = candidates_subset.apply(connectivity_score, axis=1)
                best_candidates = candidates_subset.loc[scores.sort_values(ascending=False).index]
                
                found_valid = False
                for idx, cand_row in best_candidates.iterrows():
                    if not is_fraud_target and cand_row['isFraud'] == 1:
                        continue
                    
                    G.add_edge(cand_row['nameOrig'], cand_row['nameDest'], 
                               weight=cand_row['amount'], isFraud=cand_row['isFraud'])
                    visited_tx.add(idx)
                    found_valid = True
                    break
                
                if not found_valid:
                    break 

            if G.number_of_nodes() < self.num_nodes:
                continue 
            
            if not nx.is_connected(G):
                continue
                
            edges_fraud_status = nx.get_edge_attributes(G, 'isFraud').values()
            has_fraud_edge = 1 in edges_fraud_status
            final_label = 1 if has_fraud_edge else 0
            
            if is_fraud_target and final_label == 1:
                self.graphs.append(G)
                self.labels.append(1)
                count += 1
            elif not is_fraud_target and final_label == 0:
                self.graphs.append(G)
                self.labels.append(0)
                count += 1

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
                adj[j, i] = w 
            
            degrees = dict(g.degree(weight='weight'))
            total_deg = sum(degrees.values()) + 1e-9
            for i, node in enumerate(nodes):
                adj[i, i] = degrees.get(node, 0) / total_deg

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
        
        # FIX: STABILIZED UPDATE
        # Only update a fraction of the bank (e.g., 20% max) to avoid loss spikes
        max_update = max(1, int(self.capacity * BANK_PERCENTAGE_UPDATE))
        
        if new_entries.shape[0] > max_update:
            # Pick random subset of new entries
            indices = torch.randperm(new_entries.shape[0])[:max_update]
            entries_to_add = new_entries[indices]
        else:
            entries_to_add = new_entries
            
        combined = torch.cat([self.bank, entries_to_add], dim=0)
        
        # FIFO Logic
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
            nn.Linear(self.feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
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
        
        # RETURN LOGITS (for BCEWithLogitsLoss) AND SIGMOID (for Preds) AND PHI (for Memory Bank)
        return logits, torch.sigmoid(logits), phi

# ==========================================
# 7. EXECUTION
# ==========================================

def run_experiment():
    CSV_FILE = "PS_20174392719_1491204439457_log.csv"
    try:
        dataset = PaySimDataset(CSV_FILE, total_graphs=TOTAL_GRAPHS)
    except Exception as e:
        print(f"Error: {e}")
        return

    graphs = dataset.graphs
    labels = dataset.labels
    
    idx_normal = [i for i, l in enumerate(labels) if l == 0]
    idx_fraud = [i for i, l in enumerate(labels) if l == 1]
    
    print(f"\nTotal Normal: {len(idx_normal)} | Total Fraud: {len(idx_fraud)}")
    
    if len(idx_normal) < M_VALIDATION_S_NORMAL + 5:
        print("Not enough normal graphs for validation.")
        return

    val_indices = idx_normal[:M_VALIDATION_S_NORMAL]
    remaining_normal = idx_normal[M_VALIDATION_S_NORMAL:]
    
    pool_indices = remaining_normal + idx_fraud
    np.random.shuffle(pool_indices)
    
    train_indices, test_indices = train_test_split(pool_indices, test_size=TEST_SIZE, random_state=42)
    
    print(f"Split -> Val (S_norm): {len(val_indices)} | Train: {len(train_indices)} | Test: {len(test_indices)}")

    model = QTGNN(num_nodes=NUM_NODES, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- FIX 2: WEIGHTED LOSS ---
    # Calculate pos_weight for BCEWithLogitsLoss
    # Weight = Number of Negatives (Normal) / Number of Positives (Fraud)
    # We use the ratio from the generated parameters
    n_neg = TOTAL_GRAPHS * (1 - PERCENTAGE_FRAUD)
    n_pos = TOTAL_GRAPHS * PERCENTAGE_FRAUD
    pos_weight = torch.tensor([n_neg / n_pos])
    print(f"Using Weighted Loss with pos_weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # --- TRAINING ---
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
            
            # UNPACK NEW RETURN VALUES
            logits, probs, phi = model(inputs)
            
            # USE LOGITS FOR STABLE WEIGHTED LOSS
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
                
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_indices):.4f}")

    # --- DEFINE S_normal ---
    print("\n--- Populating S_normal and Calculating Threshold ---")
    model.eval()
    S_normal_list = []
    
    with torch.no_grad():
        for i in val_indices:
            inputs = torch.tensor(dataset.get_matrix_A(graphs[i]), dtype=torch.float32)
            # Use unpacked return here too
            _, _, phi = model(inputs)
            S_normal_list.append(phi)
            
    S_normal_tensor = torch.stack(S_normal_list)
    
    # --- DYNAMIC THRESHOLD ---
    val_distances = []
    with torch.no_grad():
        for i in range(len(S_normal_tensor)):
            current_vec = S_normal_tensor[i].unsqueeze(0)
            dists = torch.norm(S_normal_tensor - current_vec, dim=1).pow(2)
            dists[dists < 1e-9] = float('inf')
            min_dist = torch.min(dists).item()
            if min_dist != float('inf'):
                val_distances.append(min_dist)
    
    if val_distances:
        mu_val = np.mean(val_distances)
        sigma_val = np.std(val_distances)
        TAU_THRESHOLD = mu_val + 2 * sigma_val 
    else:
        TAU_THRESHOLD = 0.05
        
    print(f"Dynamic Threshold (Tau): {TAU_THRESHOLD:.6f} (Mean: {mu_val:.6f}, Std: {sigma_val:.6f})")

    # --- TESTING ---
    print(f"\n--- Testing ---")
    correct = 0
    tp = 0; fp = 0; tn = 0; fn = 0
    scores_log = []
    scores_normal = []
    scores_fraud = []

    with torch.no_grad():
        for i in test_indices:
            inputs = torch.tensor(dataset.get_matrix_A(graphs[i]), dtype=torch.float32)
            label = labels[i]
            # Use unpacked return here too
            _, _, phi = model(inputs)
            
            dists = torch.norm(S_normal_tensor - phi, dim=1).pow(2)
            s_G = torch.min(dists).item()
            scores_log.append(s_G)
            
            if label == 0:
                scores_normal.append(s_G)
            else:
                scores_fraud.append(s_G)
            
            pred_class = 1 if s_G > TAU_THRESHOLD else 0
            
            if pred_class == label: correct += 1
            
            if label == 1 and pred_class == 1: tp += 1
            if label == 0 and pred_class == 1: fp += 1
            if label == 0 and pred_class == 0: tn += 1
            if label == 1 and pred_class == 0: fn += 1
            
            lbl_str = "FRAUD " if label == 1 else "NORMAL"
            
            # --- MODIFICATION: Convert phi to rounded numpy array string ---
            phi_str = np.array2string(phi.cpu().numpy(), precision=4, suppress_small=True)
            print(f"Graph {i:03d} | {lbl_str} | Score: {s_G:.6f} | Pred: {pred_class} | Phi: {phi_str}")

    acc = 100*correct/len(test_indices)
    avg_score_normal = np.mean(scores_normal) if scores_normal else 0.0
    avg_score_fraud = np.mean(scores_fraud) if scores_fraud else 0.0
    
    print(f"\nFinal Accuracy: {acc:.1f}%")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Score Stats: Max={max(scores_log):.6f}, Min={min(scores_log):.6f}, Mean={np.mean(scores_log):.6f}")
    print(f"Avg Score (Normal): {avg_score_normal:.6f}")
    print(f"Avg Score (Fraud):  {avg_score_fraud:.6f}")

if __name__ == "__main__":
    run_experiment()