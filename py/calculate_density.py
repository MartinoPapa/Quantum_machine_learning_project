import pandas as pd
import networkx as nx
import argparse
import time
from collections import Counter
import sys
import os

def analyze_degree_distribution(csv_file, chunk_size=500_000):
    """
    Loads a directed graph from CSV and computes:
    1. Total Nodes & Edges
    2. Detailed Degree Distribution (In, Out, Total)
    """
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    print(f"--- 1. Loading Graph from {csv_file} ---")
    start_time = time.time()
    
    # Using DiGraph for directed edges
    G = nx.DiGraph()
    
    try:
        # Only load necessary columns to save memory
        chunk_iter = pd.read_csv(csv_file, chunksize=chunk_size, usecols=['nameOrig', 'nameDest'])
        
        total_rows = 0
        for i, chunk in enumerate(chunk_iter):
            # Fast batch addition of edges
            # zip() creates tuples (source, dest)
            edges = list(zip(chunk['nameOrig'], chunk['nameDest']))
            G.add_edges_from(edges)
            total_rows += len(chunk)
            
            if (i+1) % 5 == 0:
                print(f"   Processed chunk {i+1} | Nodes: {G.number_of_nodes():,} | Edges: {G.number_of_edges():,}")
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    load_time = time.time() - start_time
    print(f"\nGraph Loaded in {load_time:.2f}s")
    print(f"Total Nodes: {G.number_of_nodes():,}")
    print(f"Total Edges: {G.number_of_edges():,}")
    print("-" * 50)

    # --- 2. DEGREE DISTRIBUTION ---
    print("\n--- 2. Computing Degree Distribution ---")
    
    # Calculate degrees using NetworkX iterators (memory efficient)
    # Total Degree = In-Degree + Out-Degree
    total_degrees = [d for n, d in G.degree()] 
    
    # Count frequencies
    degree_counts = Counter(total_degrees)
    
    print("\n[Total Degree Distribution (In + Out)]")
    print(f"{'Degree (k)':<12} | {'Count of Nodes':<15} | {'Percentage':<10}")
    print("-" * 45)
    
    # Sort by degree k and print the first 25
    sorted_keys = sorted(degree_counts.keys())
    for k in sorted_keys:
        count = degree_counts[k]
        perc = (count / G.number_of_nodes()) * 100
        print(f"{k:<12} | {count:<15,} | {perc:.4f}%")

    # --- 3. SUMMARY STATISTICS ---
    print("\n--- Summary ---")
    
    # Calculate nodes with degree >= 2 (potential hubs/core)
    core_nodes = sum(degree_counts[k] for k in sorted_keys if k >= 2)
    perc_core = (core_nodes / G.number_of_nodes()) * 100
    
    print(f"Max Degree:            {max(sorted_keys)}")
    print(f"Nodes with Degree = 1: {degree_counts.get(1, 0):,} ({ (degree_counts.get(1, 0)/G.number_of_nodes())*100:.2f}%)")
    print(f"Nodes with Degree >= 2:{core_nodes:,} ({perc_core:.2f}%)")
    
    if core_nodes < 200:
        print("\n[WARNING] Very few nodes have degree >= 2. Random Walk sampling will likely fail.")
        print("Suggestion: This dataset may be too sparse for topological analysis.")

    print("-" * 50)
    print(f"Analysis Complete in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Graph Nodes, Edges, and Degree Distribution.')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    
    analyze_degree_distribution(args.filename)