import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import argparse
import time
import pickle
import os

def find_pool_of_components(csv_file, output_file="pool_nodes.pkl", min_component_size=20, max_components=2000):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    print(f"--- 1. Loading & Mapping Data ---")
    start_time = time.time()
    
    # Load and map to integers
    df = pd.read_csv(csv_file, usecols=['nameOrig', 'nameDest'])
    all_nodes = pd.concat([df['nameOrig'], df['nameDest']]).unique()
    num_nodes = len(all_nodes)
    
    node_map = pd.Series(index=all_nodes, data=np.arange(num_nodes))
    src_indices = node_map[df['nameOrig']].values
    dst_indices = node_map[df['nameDest']].values
    
    # Build Sparse Matrix
    data = np.ones(len(src_indices), dtype=bool)
    adj_matrix = coo_matrix((data, (src_indices, dst_indices)), shape=(num_nodes, num_nodes))
    
    # Compute Components
    print("Computing Weakly Connected Components...")
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    
    # Analyze Sizes
    print("Analyzing component sizes...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Sort descending
    sorted_indices = np.argsort(counts)[::-1]
    
    valid_nodes_set = set()
    components_found = 0
    
    print(f"\n--- Extracting Top Components (Size >= {min_component_size}) ---")
    
    for idx in sorted_indices:
        size = counts[idx]
        if size < min_component_size:
            break # Stop if we hit components that are too small
            
        if components_found >= max_components:
            break
            
        label_id = unique_labels[idx]
        component_indices = np.where(labels == label_id)[0]
        nodes_in_comp = all_nodes[component_indices]
        
        valid_nodes_set.update(nodes_in_comp)
        components_found += 1 
        
        if components_found <= 10: # Print first 10
            print(f"Rank #{components_found}: {size} nodes")

    print(f"\nTotal Components Selected: {components_found}")
    print(f"Total Unique Nodes in Pool: {len(valid_nodes_set):,}")
    
    print(f"Saving to '{output_file}'...")
    with open(output_file, "wb") as f:
        pickle.dump(valid_nodes_set, f)
        
    print(f"Done in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Pool of Connected Components.')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    
    find_pool_of_components(args.filename)