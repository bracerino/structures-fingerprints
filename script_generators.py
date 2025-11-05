

def generate_pca_clustering_script(config):
    if config['use_2d_visualization']:
        if config['reduction_method'] == "t-SNE":
            reduction_import = "from sklearn.manifold import TSNE"
            reduction_code = f"""
# Perform t-SNE for 2D visualization
print("Performing t-SNE dimensionality reduction...")
n_samples = clustering_data.shape[0]
perplexity = min({config['tsne_perplexity']}, n_samples - 1) if n_samples > 1 else 1

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    max_iter={config['tsne_iterations']},
    learning_rate={config['tsne_learning_rate']},
    random_state=42,
    verbose=1
)
X_2d = tsne.fit_transform(clustering_data)
print(f"  t-SNE completed")
"""
        else:
            reduction_import = "import umap"
            reduction_code = f"""
# Perform UMAP for 2D visualization
print("Performing UMAP dimensionality reduction...")
reducer = umap.UMAP(
    n_components=2,
    n_neighbors={config['umap_neighbors']},
    min_dist={config['umap_min_dist']},
    random_state=42,
    verbose=True
)
X_2d = reducer.fit_transform(clustering_data)
print(f"  UMAP completed")
"""
    else:
        reduction_import = ""
        reduction_code = "\nprint('Skipping 2D visualization')\nX_2d = None"


    if config['use_pca']:
        pca_code = f"""
# =============================================================================
# PCA ANALYSIS
# =============================================================================

print(f"Performing PCA to achieve {{TARGET_VARIANCE}}% cumulative variance...")

# First, determine the number of components needed
pca_full = PCA()
pca_full.fit(X)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_ * 100)

n_components = np.argmax(cumulative_variance >= TARGET_VARIANCE) + 1

print(f"  Number of components needed: {{n_components}}")
print(f"  Cumulative variance explained: {{cumulative_variance[n_components-1]:.2f}}%")

# Perform PCA with selected number of components
pca = PCA(n_components=n_components)
clustering_data = pca.fit_transform(X)

print(f"  ✓ PCA completed: reduced from {{X.shape[1]}} to {{n_components}} dimensions")
print()
"""
        save_pca_components = """
# Add PCA components
for i in range(n_components):
    results_df[f'PC{i+1}'] = clustering_data[:, i]
"""
    else:
        pca_code = """
# =============================================================================
# SKIP PCA - USE ORIGINAL FINGERPRINTS
# =============================================================================

print("Skipping PCA - using original fingerprints for clustering...")
clustering_data = X
n_components = X.shape[1]
print(f"  Using all {n_components} original features")
print()
"""
        save_pca_components = ""

    cluster_values_str = str(config['cluster_values'])

    # Prepare visualization code
    if config['use_2d_visualization']:
        viz_save_code = """
results_df = pd.DataFrame({
    'structure': structure_names,
    'x_2d': X_2d[:, 0],
    'y_2d': X_2d[:, 1]
})
"""
        viz_plot_code = """
    # Create visualization
    print(f"\\n  Creating visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all points colored by cluster
    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=clusters,
        cmap='Set3',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    for cluster_id, info in closest_structures.items():
        idx = info['index']
        ax.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            marker='*',
            s=500,
            c='gold',
            edgecolors='black',
            linewidth=2,
            zorder=100,
            label=f'Cluster {cluster_id+1} centroid' if cluster_id < 3 else None
        )

    ax.set_xlabel('Component 1', fontsize=14)
    ax.set_ylabel('Component 2', fontsize=14)
    ax.set_title(f'K-Means Clustering (k={n_clusters}) - Centroid Structures Highlighted', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if n_clusters <= 3:
        ax.legend(fontsize=12)

    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()

    plot_path = os.path.join(k_output_folder, f'clustering_plot_k{n_clusters}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Plot saved to {plot_path}")
"""
    else:
        viz_save_code = """
results_df = pd.DataFrame({
    'structure': structure_names
})
"""
        viz_plot_code = """
    print(f"  (Skipping visualization - 2D projection not computed)")
"""

    # Description text
    if config['use_pca']:
        pca_description = f"2. Performs PCA to achieve {config['variance_threshold']}% cumulative variance"
    else:
        pca_description = "2. Uses original fingerprints (no PCA)"

    if config['use_2d_visualization']:
        viz_description = f"3. Applies {config['reduction_method']} for 2D visualization"
        step_4 = "4. Performs k-means clustering"
        step_5 = "5. Extracts structures closest to centroids"
        step_6 = "6. Organizes results into folders"
    else:
        viz_description = ""
        step_4 = "3. Performs k-means clustering"
        step_5 = "4. Extracts structures closest to centroids"
        step_6 = "5. Organizes results into folders"

    script_content = f'''#!/usr/bin/env python3
"""
Automated {'PCA + ' if config['use_pca'] else ''}K-Means Clustering Analysis Script
Generated by Structure Fingerprint Analysis Tool

This script:
1. Loads fingerprints from .npz file
{pca_description}
{viz_description}
{step_4}
{step_5}
{step_6}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
{reduction_import}
import os
import shutil
from pathlib import Path
import json

print("="*80)
print("{'PCA + ' if config['use_pca'] else ''}K-Means Clustering Analysis")
print("="*80)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

FINGERPRINTS_FILE = "{config['fingerprints_file']}"
STRUCTURES_FOLDER = "{config['structures_folder']}"
OUTPUT_BASE_FOLDER = "{config['output_base']}"
{'TARGET_VARIANCE = ' + str(config['variance_threshold']) if config['use_pca'] else '# PCA disabled - using original features'}
APPLY_STANDARDIZATION = {config['standardize']}
CLUSTER_VALUES = {cluster_values_str}
USE_PCA = {config['use_pca']}
USE_2D_VISUALIZATION = {config['use_2d_visualization']}

print(f"Configuration:")
print(f"  Fingerprints file: {{FINGERPRINTS_FILE}}")
print(f"  Structures folder: {{STRUCTURES_FOLDER}}")
print(f"  Output folder: {{OUTPUT_BASE_FOLDER}}")
{'print(f"  Target variance: {TARGET_VARIANCE}%")' if config['use_pca'] else 'print(f"  PCA: Disabled")'}
print(f"  Standardization: {{APPLY_STANDARDIZATION}}")
print(f"  Cluster values: {{CLUSTER_VALUES}}")
print(f"  2D Visualization: {{USE_2D_VISUALIZATION}}")
print()

# =============================================================================
# LOAD FINGERPRINTS
# =============================================================================

print("Loading fingerprints...")
try:
    data = np.load(FINGERPRINTS_FILE)
    fingerprints = data['fingerprints']
    structure_names = data['structure_names'].tolist()
    print(f"  ✓ Loaded {{len(fingerprints)}} fingerprints")
    print(f"  ✓ Feature dimensions: {{fingerprints.shape[1]}}")
    print(f"  ✓ Structure names: {{len(structure_names)}}")
except Exception as e:
    print(f"  ✗ Error loading fingerprints: {{str(e)}}")
    exit(1)
print()

# =============================================================================
# STANDARDIZATION (OPTIONAL)
# =============================================================================

if APPLY_STANDARDIZATION:
    print("Applying standardization...")
    scaler = StandardScaler()
    X = scaler.fit_transform(fingerprints)
    print("  ✓ Features standardized (mean=0, std=1)")
else:
    X = fingerprints
    print("Skipping standardization")
print()

{pca_code}
{reduction_code}
print()

# =============================================================================
# K-MEANS CLUSTERING FOR MULTIPLE K VALUES
# =============================================================================

# Create base output directory
os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

# Save coordinates
print("Saving coordinates...")
{viz_save_code}
{save_pca_components}

results_df.to_csv(os.path.join(OUTPUT_BASE_FOLDER, 'coordinates.csv'), index=False)
print(f"  ✓ Saved to {{os.path.join(OUTPUT_BASE_FOLDER, 'coordinates.csv')}}")
print()

print(f"Clustering on {config['use_pca'] and 'PCA-reduced' or 'original'} features ({{clustering_data.shape[1]}} dimensions)")
print()

# Perform clustering for each k value
for n_clusters in CLUSTER_VALUES:
    print(f"{{'-'*80}}")
    print(f"K-Means Clustering with {{n_clusters}} clusters")
    print(f"{{'-'*80}}")

    # Perform k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(clustering_data)
    centroids = kmeans.cluster_centers_

    print(f"  ✓ K-means completed")

    # Create output folder for this k value
    k_output_folder = os.path.join(OUTPUT_BASE_FOLDER, f"k{{n_clusters}}_clusters")
    os.makedirs(k_output_folder, exist_ok=True)

    # Find structures closest to each centroid
    closest_structures = {{}}
    centroid_info = []

    for cluster_id in range(n_clusters):
        cluster_mask = clusters == cluster_id
        cluster_points = clustering_data[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        # Find closest point to centroid
        distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
        closest_idx_in_cluster = np.argmin(distances)
        closest_idx_global = cluster_indices[closest_idx_in_cluster]

        structure_name = structure_names[closest_idx_global]
        closest_structures[cluster_id] = {{
            'index': int(closest_idx_global),
            'name': structure_name,
            'distance_to_centroid': float(distances[closest_idx_in_cluster]),
            'cluster_size': int(cluster_mask.sum())
        }}

        centroid_info.append({{
            'cluster': int(cluster_id + 1),
            'structure': structure_name,
            'cluster_size': int(cluster_mask.sum()),
            'distance_to_centroid': float(distances[closest_idx_in_cluster])
        }})

        print(f"  Cluster {{cluster_id+1}}: {{structure_name}} ({{cluster_mask.sum()}} structures)")

    # Save cluster assignments
    cluster_df = pd.DataFrame({{
        'structure': structure_names,
        'cluster': [int(c + 1) for c in clusters]  # 1-indexed for readability
    }})
    {'if X_2d is not None: cluster_df["x_2d"] = X_2d[:, 0]; cluster_df["y_2d"] = X_2d[:, 1]' if config['use_2d_visualization'] else ''}
    cluster_df.to_csv(os.path.join(k_output_folder, f'cluster_assignments_k{{n_clusters}}.csv'), index=False)

    # Save centroid information
    centroid_df = pd.DataFrame(centroid_info)
    centroid_df.to_csv(os.path.join(k_output_folder, f'centroid_structures_k{{n_clusters}}.csv'), index=False)

    # Save detailed JSON - convert all numpy types to native Python types
    clustering_info = {{
        'n_clusters': int(n_clusters),
        'n_features': int(clustering_data.shape[1]),
        'use_pca': USE_PCA,
        'use_2d_visualization': USE_2D_VISUALIZATION,
        'centroid_structures': closest_structures
    }}

    {f'clustering_info["n_components_pca"] = int(n_components)' if config['use_pca'] else ''}
    {f'clustering_info["variance_explained"] = float(cumulative_variance[n_components-1])' if config['use_pca'] else ''}

    with open(os.path.join(k_output_folder, f'clustering_info_k{{n_clusters}}.json'), 'w') as f:
        json.dump(clustering_info, f, indent=2)

    # Copy structure files closest to centroids
    centroid_structures_folder = os.path.join(k_output_folder, 'centroid_structures')
    os.makedirs(centroid_structures_folder, exist_ok=True)

    print(f"\\n  Copying centroid structures to {{centroid_structures_folder}}...")

    for cluster_id, info in closest_structures.items():
        structure_name = info['name']

        # Find the structure file
        source_path = None
        for ext in ['*.vasp', '*.poscar', '*.cif', '*.xyz', '*.pdb']:
            matches = list(Path(STRUCTURES_FOLDER).glob(f"{{structure_name}}*"))
            if matches:
                source_path = matches[0]
                break

        if source_path is None:
            # Try exact match
            potential_path = Path(STRUCTURES_FOLDER) / structure_name
            if potential_path.exists():
                source_path = potential_path

        if source_path and source_path.exists():
            dest_name = f"cluster{{cluster_id+1:02d}}_{{source_path.name}}"
            dest_path = os.path.join(centroid_structures_folder, dest_name)
            shutil.copy2(source_path, dest_path)
            print(f"    ✓ Cluster {{cluster_id+1}}: {{source_path.name}}")
        else:
            print(f"    ✗ Cluster {{cluster_id+1}}: Could not find {{structure_name}}")

{viz_plot_code}
    print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"Results saved to: {{OUTPUT_BASE_FOLDER}}")
print(f"\\nGenerated {{len(CLUSTER_VALUES)}} clustering configurations:")
for n_clusters in CLUSTER_VALUES:
    folder = os.path.join(OUTPUT_BASE_FOLDER, f"k{{n_clusters}}_clusters")
    print(f"  - {{n_clusters}} clusters: {{folder}}")
print()
print("Each folder contains:")
print("  - cluster_assignments_k*.csv: All structures with cluster assignments")
print("  - centroid_structures_k*.csv: Structures closest to centroids")
print("  - clustering_info_k*.json: Detailed clustering information")
{'print("  - clustering_plot_k*.png: Visualization")' if config['use_2d_visualization'] else ''}
print("  - centroid_structures/: Copied structure files for centroids")
print()
print("="*80)
'''

    return script_content
