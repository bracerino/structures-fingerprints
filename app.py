import streamlit as st
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from script_generators import  generate_pca_clustering_script

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

st.set_page_config(page_title="Structure Fingerprint Analysis Generator", layout="wide")

st.title("Structure Fingerprint Analysis - Script Generator")

st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
        background-color: #007acc !important; color: white !important; box-shadow: none !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
        background-color: #c82333 !important; color: white !important; box-shadow: none !important;
    }

    div.stButton > button[kind="tertiary"] {
        background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
        background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
    }

    div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

css = '''
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px !important;
}

.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f4ff !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    color: #1e3a8a !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #dbe5ff !important;
    cursor: pointer;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #e0e7ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] button:focus {
    outline: none !important;
}
</style>
'''

st.markdown(css, unsafe_allow_html=True)

main_tab1, main_tab2, tab_pca, tab_neighbor, main_tab3,  = st.tabs(["Generate Script", "Interactive Visualization: 2D t-SNE Map",
                                                      "PCA Analysis, UMAP", "Nearest Neighbor Analysis", "Getting Started Guide"])

with main_tab1:
    st.success("""
       Generate a Python script that: **Computes structure fingerprints using DScribe** • **Performs t-SNE dimensionality reduction** • **Visualizes structures in 2D space** • **Optionally colors points by energy values**
       """)

    config_tabs = st.tabs([
        "🔧 Choose Fingerprint Descriptor",
        "🗺️ t-SNE Settings",
        "🔋 Include Energies? ",
        "💾 Output Files Settings"
    ])

    with config_tabs[0]:
        st.subheader("Fingerprint Configuration")

        fingerprint_type = st.selectbox(
            "Select Fingerprint Model",
            [
                "SOAP (Smooth Overlap of Atomic Positions)",
                "ACSF (Atom-Centered Symmetry Functions)",
                "MBTR (Many-Body Tensor Representation)"
            ]
        )

        if fingerprint_type == "SOAP (Smooth Overlap of Atomic Positions)":
            st.markdown("**SOAP Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                soap_rcut = st.number_input("Cutoff radius (Å)", min_value=2.0, max_value=15.0, value=6.0, step=0.5)
                soap_nmax = st.number_input("Number of radial basis functions", min_value=1, max_value=20, value=8,
                                            step=1)
                soap_lmax = st.number_input("Maximum degree of spherical harmonics", min_value=0, max_value=15, value=6,
                                            step=1)
            with col2:
                soap_sigma = st.number_input("Gaussian width", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
                soap_periodic = st.checkbox("Periodic boundaries", value=True)
                soap_sparse = st.checkbox("Sparse output", value=False)
                soap_average = st.selectbox("Averaging method", ["off", "inner", "outer"], index=1)

        if fingerprint_type == "ACSF (Atom-Centered Symmetry Functions)":
            st.markdown("**ACSF Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                acsf_rcut = st.number_input("Cutoff radius (Å)", min_value=2.0, max_value=15.0, value=6.0, step=0.5)
                acsf_periodic = st.checkbox("Periodic boundaries", value=True)
            with col2:
                acsf_g2_params = st.text_input("G2 parameters [[eta, Rs]]", value="[[0.05, 0.0], [0.05, 2.0]]")
                acsf_g4_params = st.text_input("G4 parameters [[eta, zeta, lambda]]",
                                               value="[[0.005, 1.0, 1.0], [0.005, 2.0, -1.0]]")

        if fingerprint_type == "MBTR (Many-Body Tensor Representation)":
            st.markdown("**MBTR Parameters**")
            col1, col2 = st.columns(2)
            with col1:
                mbtr_k1 = st.checkbox("Include k=1 terms (atomic numbers)", value=True)
                mbtr_k2 = st.checkbox("Include k=2 terms (inverse distances)", value=True)
                mbtr_k3 = st.checkbox("Include k=3 terms (cosines of angles)", value=True)
                mbtr_periodic = st.checkbox("Periodic boundaries", value=True)
            with col2:
                mbtr_grid_k2_min = st.number_input("k2 grid min", value=0.0)
                mbtr_grid_k2_max = st.number_input("k2 grid max", value=1.0)
                mbtr_grid_k2_sigma = st.number_input("k2 sigma", value=0.1)

    with config_tabs[1]:
        st.subheader("t-SNE Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            tsne_perplexity = st.number_input("Perplexity", min_value=5, max_value=50, value=30, step=5)

        with col2:
            tsne_iterations = st.number_input("Max iterations", min_value=250, max_value=5000, value=1000, step=250)

        with col3:
            tsne_learning_rate = st.number_input("Learning rate", min_value=10.0, max_value=1000.0, value=200.0,
                                                 step=50.0)

    with config_tabs[2]:
        st.subheader("Energy Coloring")

        use_energy_coloring = st.checkbox("Color points by energy values", value=True)

        if use_energy_coloring:
            st.info("The script will look for 'energies.txt' file with structure names and energies")
            col1, col2 = st.columns(2)
            with col1:
                energy_filename = st.text_input("Energy file name", value="energies.txt")
            with col2:
                colormap = st.selectbox("Colormap",
                                        ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdYlBu_r"],
                                        index=0)

    with config_tabs[3]:
        st.subheader("Output Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Plot Settings**")
            output_plot_name = st.text_input("Output plot filename", value="tsne_fingerprints.png")
            output_dpi = st.number_input("Figure DPI", min_value=100, max_value=600, value=300, step=50)
            figure_width = st.number_input("Figure width (inches)", min_value=6, max_value=20, value=12, step=1)
            figure_height = st.number_input("Figure height (inches)", min_value=6, max_value=20, value=10, step=1)

        with col2:
            st.markdown("**Data Export Settings**")
            save_fingerprints = st.checkbox("Save computed fingerprints to file", value=True)
            if save_fingerprints:
                fingerprints_filename = st.text_input("Fingerprints output filename", value="fingerprints.npy")

            save_tsne_coords = st.checkbox("Save t-SNE coordinates", value=True)
            if save_tsne_coords:
                tsne_coords_filename = st.text_input("t-SNE coordinates filename", value="tsne_coordinates.csv")


    def generate_fingerprint_code(fp_type):
        if fp_type == "SOAP (Smooth Overlap of Atomic Positions)":
            avg_method = soap_average
            return f"""
from dscribe.descriptors import SOAP

soap = SOAP(
    species=species,
    periodic={soap_periodic},
    r_cut={soap_rcut},
    n_max={soap_nmax},
    l_max={soap_lmax},
    sigma={soap_sigma},
    average="{avg_method}",
    sparse={soap_sparse}
)

for i, (name, atoms) in enumerate(structures):
    print(f"  Computing SOAP for {{name}} ({{i+1}}/{{len(structures)}})")
    fingerprint = soap.create(atoms)
    if "{avg_method}" != "off":
        fingerprints.append(fingerprint)
    else:
        fingerprints.append(fingerprint.mean(axis=0))
    structure_names.append(name)
"""

        elif fp_type == "ACSF (Atom-Centered Symmetry Functions)":
            return f"""
from dscribe.descriptors import ACSF
import numpy as np

g2_params = np.array({acsf_g2_params})
g4_params = np.array({acsf_g4_params})

acsf = ACSF(
    species=species,
    r_cut={acsf_rcut},
    g2_params=g2_params,
    g4_params=g4_params,
    periodic={acsf_periodic}
)

for i, (name, atoms) in enumerate(structures):
    print(f"  Computing ACSF for {{name}} ({{i+1}}/{{len(structures)}})")
    fingerprint = acsf.create(atoms)
    fingerprints.append(fingerprint.mean(axis=0))
    structure_names.append(name)
"""

        elif fp_type == "MBTR (Many-Body Tensor Representation)":
            k1_dict = f'{{"geometry": {{"function": "atomic_number"}}, "grid": {{"min": 0, "max": 100, "n": 100, "sigma": 0.1}}}}' if mbtr_k1 else 'None'
            k2_dict = f'{{"geometry": {{"function": "inverse_distance"}}, "grid": {{"min": {mbtr_grid_k2_min}, "max": {mbtr_grid_k2_max}, "n": 100, "sigma": {mbtr_grid_k2_sigma}}}, "weighting": {{"function": "exp", "scale": 0.5, "threshold": 1e-3}}}}' if mbtr_k2 else 'None'
            k3_dict = f'{{"geometry": {{"function": "cosine"}}, "grid": {{"min": -1, "max": 1, "n": 100, "sigma": 0.1}}, "weighting": {{"function": "exp", "scale": 0.5, "threshold": 1e-3}}}}' if mbtr_k3 else 'None'

            return f"""
from dscribe.descriptors import MBTR

mbtr = MBTR(
    species=species,
    k1={k1_dict},
    k2={k2_dict},
    k3={k3_dict},
    periodic={mbtr_periodic},
    normalization="l2"
)

for i, (name, atoms) in enumerate(structures):
    print(f"  Computing MBTR for {{name}} ({{i+1}}/{{len(structures)}})")
    fingerprint = mbtr.create(atoms)
    fingerprints.append(fingerprint)
    structure_names.append(name)
"""


    def generate_script():
        energy_code = ""
        if use_energy_coloring:
            energy_code = f"""
print("Loading energy values...")
energies = {{}}
try:
    with open("{energy_filename}", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    energy = float(parts[1])
                    energies[name] = energy
    print(f"  Loaded energies for {{len(energies)}} structures")
except FileNotFoundError:
    print(f"  Warning: Energy file '{energy_filename}' not found. Proceeding without energy coloring.")
    energies = None
except Exception as e:
    print(f"  Warning: Could not load energies: {{str(e)}}")
    energies = None
"""
        else:
            energy_code = "\nenergies = None\n"

        save_fingerprints_code = ""
        if save_fingerprints:
            npz_filename = fingerprints_filename.replace('.npy', '.npz')
            save_fingerprints_code = f"""
print("Saving fingerprints, structure names, and energies...")
# Prepare energy array matching structure names
energy_array = np.array([energies.get(name, np.nan) if energies else np.nan for name in structure_names])

np.savez("{npz_filename}", 
         fingerprints=np.array(fingerprints),
         structure_names=np.array(structure_names),
         energies=energy_array)
print(f"  Fingerprints, structure names, and energies saved to {npz_filename}")
"""
        save_tsne_code = ""
        if save_tsne_coords:
            save_tsne_code = f"""
print("Saving t-SNE coordinates...")
tsne_df = pd.DataFrame({{
    "structure": structure_names,
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1]
}})
if energies:
    tsne_df["energy"] = [energies.get(name, np.nan) for name in structure_names]
tsne_df.to_csv("{tsne_coords_filename}", index=False)
print(f"  t-SNE coordinates saved to {tsne_coords_filename}")
"""

        plot_code = ""
        if use_energy_coloring:
            plot_code = f"""
if energies:
    energy_values = [energies.get(name, np.nan) for name in structure_names]
    valid_mask = ~np.isnan(energy_values)

    if np.any(valid_mask):
        scatter = ax.scatter(
            X_tsne[valid_mask, 0], 
            X_tsne[valid_mask, 1],
            c=np.array(energy_values)[valid_mask],
            cmap="{colormap}",
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Energy (eV)", fontsize=14)

        if np.any(~valid_mask):
            ax.scatter(
                X_tsne[~valid_mask, 0],
                X_tsne[~valid_mask, 1],
                c="gray",
                s=100,
                alpha=0.5,
                edgecolors="black",
                linewidth=0.5,
                label="No energy"
            )
            ax.legend(fontsize=12)
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
else:
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
"""
        else:
            plot_code = """
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100, alpha=0.7, edgecolors="black", linewidth=0.5)
"""

        fingerprint_code = generate_fingerprint_code(fingerprint_type)

        poscar_list = '["POSCAR", "CONTCAR"] + [f"POSCAR_{{i}}" for i in range(1, 100)]'

        script = f"""#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ase.io import read
import os
import glob
import pandas as pd

print("="*60)
print("Structure Fingerprint Analysis with t-SNE")
print("="*60)
print()

structures_folder = "structures"

print(f"Loading structures from '{{structures_folder}}' folder...")

structure_files = glob.glob(os.path.join(structures_folder, "*"))
structure_files = [f for f in structure_files if os.path.isfile(f)]

valid_extensions = [".vasp", ".poscar", ".cif", ".xyz", ".pdb", ".lmp", ".xsf", ".pw"]
poscar_names = {poscar_list}
structure_files = [f for f in structure_files if 
    any(f.lower().endswith(ext) for ext in valid_extensions) or
    os.path.basename(f) in poscar_names or
    os.path.basename(f).startswith("POSCAR")
]

print(f"  Found {{len(structure_files)}} structure files")

if len(structure_files) == 0:
    print("  Error: No structure files found!")
    exit(1)

structures = []
for filepath in structure_files:
    try:
        atoms = read(filepath)
        name = os.path.basename(filepath)
        structures.append((name, atoms))
    except Exception as e:
        print(f"  Warning: Could not read {{filepath}}: {{str(e)}}")

print(f"  Successfully loaded {{len(structures)}} structures")
print()

{energy_code}

print("Determining unique species across all structures...")
species = set()
for name, atoms in structures:
    species.update(atoms.get_chemical_symbols())
species = sorted(list(species))
print(f"  Found {{len(species)}} unique elements: {{', '.join(species)}}")
print()

print("Computing fingerprints...")
fingerprints = []
structure_names = []
{fingerprint_code}

print(f"  Computed {{len(fingerprints)}} fingerprints")
print()

{save_fingerprints_code}

print("Performing t-SNE dimensionality reduction...")
X = np.array(fingerprints)

n_samples = X.shape[0]

if n_samples == 1:
    print("  Warning: Only 1 sample found. Cannot perform t-SNE (requires at least 2 samples).")
    print("  Skipping t-SNE and creating simple plot...")

    X_tsne = np.array([[0, 0]])

elif n_samples == 2:
    print("  Warning: Only 2 samples found. t-SNE will place them on a line.")
    perplexity = 1
    print(f"  Using perplexity={{perplexity}}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter={tsne_iterations},
        learning_rate={tsne_learning_rate},
        random_state=42,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X)

else:
    requested_perplexity = {tsne_perplexity}

    if n_samples <= requested_perplexity:
        perplexity = min(max(1, n_samples - 1), 30)
        print(f"  Adjusting perplexity from {{requested_perplexity}} to {{perplexity}} (n_samples={{n_samples}})")
    else:
        perplexity = requested_perplexity
        print(f"  Using perplexity={{perplexity}} (n_samples={{n_samples}})")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter={tsne_iterations},
        learning_rate={tsne_learning_rate},
        random_state=42,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X)

print("  t-SNE completed")
print()

{save_tsne_code}

print("Creating visualization...")
fig, ax = plt.subplots(figsize=({figure_width}, {figure_height}))

{plot_code}

for i, name in enumerate(structure_names):
    ax.annotate(
        name,
        (X_tsne[i, 0], X_tsne[i, 1]),
        fontsize=8,
        alpha=0.7,
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlabel("t-SNE Component 1", fontsize=14)
ax.set_ylabel("t-SNE Component 2", fontsize=14)
ax.set_title("Structure Fingerprints - t-SNE Visualization", fontsize=16, fontweight="bold")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("{output_plot_name}", dpi={output_dpi}, bbox_inches="tight")
print(f"  Plot saved to {output_plot_name}")
print()

print("="*60)
print("Analysis completed successfully!")
print("="*60)
"""

        return script


    if st.button("Generate Python Script", type="primary"):
        script_content = generate_script()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_filename = f"fingerprint_analysis_{timestamp}.py"

        st.success(f"Script generated successfully!")

        st.download_button(
            label="Download Python Script",
            data=script_content,
            file_name=script_filename,
            mime="text/x-python",
            type="primary"
        )

        with st.expander("View Generated Script", expanded=True):
            st.code(script_content, language="python")
with main_tab2:
    st.header("Interactive t-SNE Visualization - Upload CSV file from the generated script")

    with st.expander("ℹ️ Expected File Format"):
        st.markdown("""
        **Required CSV Structure:**

        The uploaded CSV file should contain the following columns:

        - `structure` - Name of each structure (string)
        - `x` - t-SNE x-coordinate (float)
        - `y` - t-SNE y-coordinate (float)
        - `energy` - Energy value for coloring (float, optional)

        **Example:**
        ```
        structure,x,y,energy
        POSCAR_1,-15.234,8.567,-123.45
        POSCAR_2,12.891,-4.123,-118.92
        POSCAR_3,3.456,15.789,-125.67
        ```

        **Notes:**
        - The file is automatically generated by the Python script from the "Generate Script" tab
        - If the `energy` column is present, you can color points by energy values
        - Column names must match exactly (case-sensitive)
        """)

    tsne_file = st.file_uploader("Upload t-SNE coordinates CSV", type=['csv'])

    if tsne_file is not None:
        df = pd.read_csv(tsne_file)

        viz_col1, viz_col2 = st.columns([2, 1])

        with viz_col2:
            st.subheader("Plot Settings")

            point_size = st.slider("Point size", min_value=5, max_value=50, value=25, step=5)

            show_labels = st.checkbox("Show structure labels", value=False)

            if 'energy' in df.columns and df['energy'].notna().any():
                st.markdown("---")
                st.subheader("Clustering Options")

                perform_clustering = st.checkbox("Perform k-means clustering", value=False)

                if perform_clustering:
                    n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=3, step=1)

                    clustering_feature = st.radio(
                        "Cluster based on:",
                        ["Energy values", "t-SNE coordinates"]
                    )

                    if clustering_feature == "Energy values":
                        energy_data = df[['energy']].dropna()
                        valid_indices = df['energy'].notna()
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = np.full(len(df), -1)
                        clusters[valid_indices] = kmeans.fit_predict(energy_data)
                        df['cluster'] = clusters
                    else:
                        tsne_coords = df[['x', 'y']].values
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        df['cluster'] = kmeans.fit_predict(tsne_coords)

                    color_by_cluster = st.checkbox("Color by cluster groups", value=True)
                else:
                    color_by_cluster = False

                st.markdown("---")

                if not perform_clustering or not color_by_cluster:
                    color_by_energy = st.checkbox("Color by energy", value=True)
                    if color_by_energy:
                        plot_colormap = st.selectbox(
                            "Colormap",
                            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "RdYlBu_r", "Spectral_r"],
                            index=0
                        )

                        st.markdown("**Color Scale Range**")
                        energy_min = float(df['energy'].min())
                        energy_max = float(df['energy'].max())

                        use_custom_range = st.checkbox("Use custom range", value=False)

                        if use_custom_range:
                            color_min = st.number_input("Minimum", value=energy_min, step=0.01)
                            color_max = st.number_input("Maximum", value=energy_max, step=0.01)
                        else:
                            color_min = energy_min
                            color_max = energy_max
                else:
                    color_by_energy = False
            else:
                color_by_energy = False
                perform_clustering = False
                color_by_cluster = False
                st.info("No energy data found in file")

        with viz_col1:
            st.subheader("t-SNE Scatter Plot")

            if perform_clustering and color_by_cluster:
                df['cluster_label'] = df['cluster'].astype(str)
                df.loc[df['cluster'] == -1, 'cluster_label'] = 'No data'

                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    color='cluster_label',
                    hover_data=['structure', 'energy', 'cluster'],
                    labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'cluster_label': 'Cluster'},
                    title='Interactive t-SNE Visualization - K-Means Clustering',
                    width=800,
                    height=600,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_traces(marker=dict(size=point_size, line=dict(width=1, color='DarkSlateGrey')))

            elif color_by_energy and 'energy' in df.columns:
                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    color='energy',
                    hover_data=['structure', 'energy'],
                    labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'energy': 'Energy (eV)'},
                    title='Interactive t-SNE Visualization',
                    color_continuous_scale=plot_colormap.lower(),
                    range_color=[color_min, color_max],
                    width=800,
                    height=600
                )

                fig.update_traces(marker=dict(size=point_size, line=dict(width=1, color='DarkSlateGrey')))

            else:
                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    hover_data=['structure'],
                    labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
                    title='Interactive t-SNE Visualization',
                    width=800,
                    height=600
                )

                fig.update_traces(
                    marker=dict(size=point_size, color='steelblue', line=dict(width=1, color='DarkSlateGrey')))

            if show_labels:
                for i, row in df.iterrows():
                    fig.add_annotation(
                        x=row['x'],
                        y=row['y'],
                        text=row['structure'],
                        showarrow=False,
                        font=dict(size=12),
                        xshift=5,
                        yshift=5,
                        opacity=0.8
                    )

            fig.update_layout(
                hovermode='closest',
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    title_font=dict(size=24, color='black'),
                    tickfont=dict(size=18, color='black'),
                    linecolor='black',
                    linewidth=2
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    title_font=dict(size=24, color='black'),
                    tickfont=dict(size=18, color='black'),
                    linecolor='black',
                    linewidth=2
                ),
                hoverlabel=dict(
                    font_size=18,
                    font_family="Arial"
                ),
                coloraxis_colorbar=dict(
                    title_font=dict(size=20, color='black'),
                    tickfont=dict(size=16, color='black'),
                    thickness=20,
                    len=0.7
                )
            )

            st.plotly_chart(fig, )

        st.subheader("Data Preview")
        st.dataframe(df.head(), width="stretch")

        st.subheader("Export Options")

        html_str = fig.to_html()
        st.download_button(
            label="Download Interactive Plot (HTML)",
            data=html_str,
            file_name="tsne_interactive.html",
            mime="text/html"
        )
with tab_pca:
    st.header("Principal Component Analysis (PCA) - Upload Fingerprints")

    with st.expander("ℹ️ Expected File Format"):
        st.markdown("""
        **Required File Format:**

        - Upload the `.npy` file generated by the Python script (e.g., `fingerprints.npy`)
        - This file contains the computed fingerprint vectors for all structures
        - Optionally upload a CSV file with structure names and energies for additional analysis

        **What PCA does:**
        - Reduces dimensionality of fingerprint data
        - Shows how much variance each principal component explains
        - Helps understand which features are most important
        - Can reveal patterns in your structural data
        """)

    fingerprint_tabs = st.tabs([
        "📊 PCA Analysis",
        "🔧 Generate Analysis Script",
        "🧮 Compute t-SNE from Fingerprints",
    ])
    with fingerprint_tabs[1]:
        st.subheader("🔧 Generate Automated Clustering Script")

        st.markdown("""
        Generate a Python script that will:
        - Load fingerprints from `.npz` file
        - Optionally apply PCA dimensionality reduction
        - Optionally create 2D visualization (t-SNE/UMAP)
        - Perform k-means clustering
        - Extract structures closest to centroids
        - Organize results into folders
        """)

        st.markdown("---")

        script_col1, script_col2 = st.columns(2)

        with script_col1:
            st.markdown("### Feature Reduction Configuration")

            script_use_pca = st.checkbox(
                "Apply PCA dimensionality reduction",
                value=True,
                key='script_use_pca',
                help="If disabled, clustering will be performed on original fingerprints"
            )

            if script_use_pca:
                script_variance_threshold = st.slider(
                    "Target cumulative variance (%)",
                    min_value=80,
                    max_value=99,
                    value=95,
                    step=1,
                    key='script_var_threshold',
                    help="PCA will use enough components to explain this % of variance"
                )
            else:
                st.info("ℹ️ Clustering will use all original fingerprint features")

            script_standardize = st.checkbox(
                "Apply standardization",
                value=True,
                key='script_standardize',
                help="Standardize features to mean=0, std=1"
            )

            st.markdown("---")
            st.markdown("### 2D Visualization (Optional)")

            script_use_2d = st.checkbox(
                "Create 2D visualization plots",
                value=True,
                key='script_use_2d',
                help="Generate t-SNE or UMAP 2D plots (optional, does not affect clustering)"
            )

            if script_use_2d:
                script_reduction_methods = ["t-SNE"]
                if UMAP_AVAILABLE:
                    script_reduction_methods.append("UMAP")

                script_reduction_method = st.selectbox(
                    "2D visualization method",
                    options=script_reduction_methods,
                    index=0,
                    key='script_reduction_method'
                )

                if script_reduction_method == "t-SNE":
                    st.markdown("**t-SNE Parameters**")
                    script_tsne_perp = st.number_input("Perplexity", min_value=5, max_value=50, value=30,
                                                       key='script_tsne_perp')
                    script_tsne_iter = st.number_input("Max iterations", min_value=250, max_value=5000,
                                                       value=1000, step=250, key='script_tsne_iter')
                    script_tsne_lr = st.number_input("Learning rate", min_value=10.0, max_value=1000.0,
                                                     value=200.0, step=50.0, key='script_tsne_lr')
                else:
                    st.markdown("**UMAP Parameters**")
                    script_umap_neighbors = st.number_input("N neighbors", min_value=2, max_value=200, value=15,
                                                            step=5, key='script_umap_neighbors')
                    script_umap_min_dist = st.number_input("Min distance", min_value=0.0, max_value=1.0,
                                                           value=0.1, step=0.05, key='script_umap_min_dist')
            else:
                st.info("ℹ️ No 2D plots will be generated (faster execution)")

        with script_col2:
            st.markdown("### K-Means Clustering Configuration")

            st.info(
                f"Clustering will be performed on: **{'PCA-reduced features' if script_use_pca else 'Original fingerprints'}**")

            st.markdown("**Number of Clusters**")

            script_cluster_mode = st.radio(
                "Cluster configuration:",
                ["Single value", "Range with step"],
                index=0,
                key='script_cluster_mode'
            )

            if script_cluster_mode == "Single value":
                script_n_clusters = st.number_input(
                    "Number of clusters",
                    min_value=2,
                    max_value=20,
                    value=3,
                    step=1,
                    key='script_n_clusters_single'
                )
                cluster_values = [script_n_clusters]
            else:
                cluster_col1, cluster_col2, cluster_col3 = st.columns(3)
                with cluster_col1:
                    script_cluster_min = st.number_input(
                        "Min clusters",
                        min_value=2,
                        max_value=19998,
                        value=2,
                        step=1,
                        key='script_cluster_min'
                    )
                with cluster_col2:
                    script_cluster_max = st.number_input(
                        "Max clusters",
                        min_value=2,
                        max_value=20000,
                        value=6,
                        step=1,
                        key='script_cluster_max'
                    )
                with cluster_col3:
                    script_cluster_step = st.number_input(
                        "Step",
                        min_value=1,
                        max_value=10000,
                        value=1,
                        step=1,
                        key='script_cluster_step'
                    )

                cluster_values = list(range(script_cluster_min, script_cluster_max + 1, script_cluster_step))
                st.info(f"Will test: {cluster_values} clusters")

            st.markdown("---")
            st.markdown("### File Paths")

            script_fingerprints_file = st.text_input(
                "Fingerprints file (.npz)",
                value="fingerprints.npz",
                key='script_fp_file',
                help="Path to the .npz file containing fingerprints and structure names"
            )

            script_structures_folder = st.text_input(
                "Structures folder",
                value="structures",
                key='script_struct_folder',
                help="Folder containing the original structure files"
            )

            script_output_base = st.text_input(
                "Output base folder",
                value="clustering_results",
                key='script_output_base',
                help="Base folder for all results"
            )

        st.markdown("---")

        if st.button("🔧 Generate Analysis Script", type="primary", key='gen_pca_script'):
            # Prepare configuration dictionary
            config = {
                'use_pca': script_use_pca,
                'standardize': script_standardize,
                'use_2d_visualization': script_use_2d,
                'cluster_values': cluster_values,
                'fingerprints_file': script_fingerprints_file,
                'structures_folder': script_structures_folder,
                'output_base': script_output_base
            }

            if script_use_pca:
                config['variance_threshold'] = script_variance_threshold

            if script_use_2d:
                config['reduction_method'] = script_reduction_method
                if script_reduction_method == "t-SNE":
                    config['tsne_perplexity'] = script_tsne_perp
                    config['tsne_iterations'] = script_tsne_iter
                    config['tsne_learning_rate'] = script_tsne_lr
                else:
                    config['umap_neighbors'] = script_umap_neighbors
                    config['umap_min_dist'] = script_umap_min_dist

            # Generate the script
            script_content = generate_pca_clustering_script(config)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_filename = f"clustering_analysis_{timestamp}.py"

            st.success("✅ Script generated successfully!")

            st.download_button(
                label="📥 Download Python Script",
                data=script_content,
                file_name=script_filename,
                mime="text/x-python",
                type="primary"
            )

            with st.expander("📄 View Generated Script", expanded=True):
                st.code(script_content, language="python")

            st.markdown("---")
            st.markdown("### 📋 Usage Instructions")

            required_packages = "numpy pandas matplotlib scikit-learn"
            if script_use_2d and script_reduction_method == "UMAP":
                required_packages += " umap-learn"

            workflow_steps = []
            workflow_steps.append(f"- Load fingerprints from `{script_fingerprints_file}`")
            if script_use_pca:
                workflow_steps.append(f"- Apply PCA to achieve **{script_variance_threshold}% variance**")
            else:
                workflow_steps.append("- Use **original fingerprints** (no PCA)")
            if script_use_2d:
                workflow_steps.append(f"- Create 2D visualization using **{script_reduction_method}**")
            workflow_steps.append(f"- Perform k-means clustering for: **{cluster_values}**")
            workflow_steps.append("- Extract and copy centroid structures")

            st.markdown(f"""
            **Before running the script:**

            1. Ensure you have the required packages:
    ```bash
            pip install {required_packages}
    ```

            2. Make sure you have:
               - `{script_fingerprints_file}` in the same directory
               - `{script_structures_folder}/` folder with your structure files

            3. Run the script:
    ```bash
            python {script_filename}
    ```

            **What the script will do:**

            {chr(10).join(workflow_steps)}

            **Output structure:**
    ```
            {script_output_base}/
            ├── coordinates.csv
            ├── k{cluster_values[0]}_clusters/
            │   ├── cluster_assignments_k{cluster_values[0]}.csv
            │   ├── centroid_structures_k{cluster_values[0]}.csv
            │   ├── clustering_info_k{cluster_values[0]}.json
            {'│   ├── clustering_plot_k' + str(cluster_values[0]) + '.png' if script_use_2d else ''}
            │   └── centroid_structures/
            │       ├── cluster01_structure.vasp
            │       ├── cluster02_structure.vasp
            │       └── ...
            {'├── k' + str(cluster_values[1]) + '_clusters/' if len(cluster_values) > 1 else ''}
            {'│   └── ...' if len(cluster_values) > 1 else ''}
    ```
            """)
    with fingerprint_tabs[2]:
        st.subheader("Compute t-SNE Directly from Fingerprints")

        st.info("Upload your fingerprints.npy file to automatically compute t-SNE and visualize the results")

        fingerprint_file_tsne = st.file_uploader("Upload fingerprints file (.npy or .npz)",
                                                 type=['npy', 'npz'],
                                                 key='tsne_fingerprints')

        if fingerprint_file_tsne is not None:
            from sklearn.manifold import TSNE

            if fingerprint_file_tsne.name.endswith('.npz'):
                data = np.load(fingerprint_file_tsne)
                fingerprints_data = data['fingerprints']
                structure_names_tsne = data['structure_names'].tolist() if 'structure_names' in data else None
                if 'energies' in data:
                    energies_tsne = data['energies']
                    valid_energy_count = np.sum(~np.isnan(energies_tsne))
                    st.success(f"✅ Energies loaded from .npz file: {valid_energy_count} structures with energy data")
            else:
                fingerprints_data = np.load(fingerprint_file_tsne)
                structure_names_tsne = None

            st.success(
                f"✅ Loaded fingerprints: {fingerprints_data.shape[0]} structures, {fingerprints_data.shape[1]} features")

            energies_tsne = None

            if structure_names_tsne is not None:
                st.success(f"✅ Structure names loaded from .npz file: {len(structure_names_tsne)} structures")

            coord_file_tsne = st.file_uploader(
                "Upload CSV with energies (optional - names loaded from .npz if available)",
                type=['csv'],
                key='tsne_csv_upload'
            )

            if coord_file_tsne is not None:
                coord_df_tsne = pd.read_csv(coord_file_tsne)
                if structure_names_tsne is None and 'structure' in coord_df_tsne.columns:
                    structure_names_tsne = coord_df_tsne['structure'].tolist()

                if 'energy' in coord_df_tsne.columns:
                    energies_tsne = coord_df_tsne['energy'].values

            st.markdown("---")
            st.subheader("t-SNE Configuration")

            tsne_col1, tsne_col2, tsne_col3 = st.columns(3)

            with tsne_col1:
                tsne_perplexity_calc = st.number_input("Perplexity", min_value=5, max_value=50, value=30, step=5,
                                                       key='tsne_perp_calc')

            with tsne_col2:
                tsne_iterations_calc = st.number_input("Max iterations", min_value=250, max_value=5000, value=1000,
                                                       step=250, key='tsne_iter_calc')

            with tsne_col3:
                tsne_learning_rate_calc = st.number_input("Learning rate", min_value=10.0, max_value=1000.0,
                                                          value=200.0, step=50.0, key='tsne_lr_calc')

            if st.button("Compute t-SNE", type="primary", key='compute_tsne_btn'):
                with st.spinner("Computing t-SNE... This may take a moment"):
                    n_samples = fingerprints_data.shape[0]

                    if n_samples <= tsne_perplexity_calc:
                        actual_perplexity = min(max(1, n_samples - 1), 30)
                        st.warning(
                            f"Adjusted perplexity from {tsne_perplexity_calc} to {actual_perplexity} (n_samples={n_samples})")
                    else:
                        actual_perplexity = tsne_perplexity_calc

                    tsne_model = TSNE(
                        n_components=2,
                        perplexity=actual_perplexity,
                        max_iter=int(tsne_iterations_calc),
                        learning_rate=tsne_learning_rate_calc,
                        random_state=42,
                        verbose=1
                    )

                    X_tsne_computed = tsne_model.fit_transform(fingerprints_data)

                    tsne_computed_df = pd.DataFrame({
                        'x': X_tsne_computed[:, 0],
                        'y': X_tsne_computed[:, 1]
                    })

                    if structure_names_tsne is not None and len(structure_names_tsne) == len(X_tsne_computed):
                        tsne_computed_df['structure'] = structure_names_tsne
                    else:
                        tsne_computed_df['structure'] = [f'Structure_{i + 1}' for i in range(len(X_tsne_computed))]

                    if energies_tsne is not None and len(energies_tsne) == len(X_tsne_computed):
                        tsne_computed_df['energy'] = energies_tsne

                    st.session_state['tsne_computed_df'] = tsne_computed_df

                    st.success("✅ t-SNE computation completed!")

            if 'tsne_computed_df' in st.session_state:
                df_viz = st.session_state['tsne_computed_df']

                st.markdown("---")
                st.subheader("Interactive t-SNE Visualization")

                viz_col1, viz_col2 = st.columns([2, 1])

                with viz_col2:
                    st.markdown("**Plot Settings**")

                    point_size_calc = st.slider("Point size", min_value=5, max_value=50, value=25, step=5,
                                                key='point_size_calc')

                    show_labels_calc = st.checkbox("Show structure labels", value=False, key='show_labels_calc')

                    if 'energy' in df_viz.columns and df_viz['energy'].notna().any():
                        st.markdown("---")
                        st.subheader("Clustering Options")

                        perform_clustering_calc = st.checkbox("Perform k-means clustering", value=False,
                                                              key='cluster_calc')

                        if perform_clustering_calc:
                            n_clusters_calc = st.slider("Number of clusters", min_value=2, max_value=20, value=3,
                                                        step=1, key='n_clusters_calc')

                            clustering_feature_calc = st.radio(
                                "Cluster based on:",
                                ["Energy values", "t-SNE coordinates"],
                                key='cluster_feature_calc'
                            )

                            if clustering_feature_calc == "Energy values":
                                energy_data = df_viz[['energy']].dropna()
                                valid_indices = df_viz['energy'].notna()
                                kmeans = KMeans(n_clusters=n_clusters_calc, random_state=42, n_init=10)
                                clusters = np.full(len(df_viz), -1)
                                clusters[valid_indices] = kmeans.fit_predict(energy_data)
                                df_viz['cluster'] = clusters
                            else:
                                tsne_coords = df_viz[['x', 'y']].values
                                kmeans = KMeans(n_clusters=n_clusters_calc, random_state=42, n_init=10)
                                df_viz['cluster'] = kmeans.fit_predict(tsne_coords)

                            color_by_cluster_calc = st.checkbox("Color by cluster groups", value=True,
                                                                key='color_cluster_calc')
                        else:
                            color_by_cluster_calc = False

                        st.markdown("---")

                        if not perform_clustering_calc or not color_by_cluster_calc:
                            color_by_energy_calc = st.checkbox("Color by energy", value=True, key='color_energy_calc')
                            if color_by_energy_calc:
                                plot_colormap_calc = st.selectbox(
                                    "Colormap",
                                    ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "RdYlBu_r",
                                     "Spectral_r"],
                                    index=0,
                                    key='colormap_calc'
                                )

                                st.markdown("**Color Scale Range**")
                                energy_min_calc = float(df_viz['energy'].min())
                                energy_max_calc = float(df_viz['energy'].max())

                                use_custom_range_calc = st.checkbox("Use custom range", value=False,
                                                                    key='custom_range_calc')

                                if use_custom_range_calc:
                                    color_min_calc = st.number_input("Minimum", value=energy_min_calc, step=0.01,
                                                                     key='color_min_calc')
                                    color_max_calc = st.number_input("Maximum", value=energy_max_calc, step=0.01,
                                                                     key='color_max_calc')
                                else:
                                    color_min_calc = energy_min_calc
                                    color_max_calc = energy_max_calc
                        else:
                            color_by_energy_calc = False
                    else:
                        color_by_energy_calc = False
                        perform_clustering_calc = False
                        color_by_cluster_calc = False
                        st.info("No energy data available for coloring")

                with viz_col1:
                    if perform_clustering_calc and color_by_cluster_calc:
                        df_viz['cluster_label'] = df_viz['cluster'].astype(str)
                        df_viz.loc[df_viz['cluster'] == -1, 'cluster_label'] = 'No data'

                        fig_calc = px.scatter(
                            df_viz,
                            x='x',
                            y='y',
                            color='cluster_label',
                            hover_data=['structure', 'energy', 'cluster'] if 'energy' in df_viz.columns else [
                                'structure', 'cluster'],
                            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'cluster_label': 'Cluster'},
                            title='t-SNE Visualization - K-Means Clustering',
                            width=800,
                            height=600,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )

                        fig_calc.update_traces(
                            marker=dict(size=point_size_calc, line=dict(width=1, color='DarkSlateGrey')))

                    elif color_by_energy_calc and 'energy' in df_viz.columns:
                        fig_calc = px.scatter(
                            df_viz,
                            x='x',
                            y='y',
                            color='energy',
                            hover_data=['structure', 'energy'],
                            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'energy': 'Energy (eV)'},
                            title='t-SNE Visualization',
                            color_continuous_scale=plot_colormap_calc.lower(),
                            range_color=[color_min_calc, color_max_calc],
                            width=800,
                            height=600
                        )

                        fig_calc.update_traces(
                            marker=dict(size=point_size_calc, line=dict(width=1, color='DarkSlateGrey')))

                    else:
                        fig_calc = px.scatter(
                            df_viz,
                            x='x',
                            y='y',
                            hover_data=['structure'],
                            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
                            title='t-SNE Visualization',
                            width=800,
                            height=600
                        )

                        fig_calc.update_traces(
                            marker=dict(size=point_size_calc, color='steelblue',
                                        line=dict(width=1, color='DarkSlateGrey')))

                    if show_labels_calc:
                        for i, row in df_viz.iterrows():
                            fig_calc.add_annotation(
                                x=row['x'],
                                y=row['y'],
                                text=row['structure'],
                                showarrow=False,
                                font=dict(size=12),
                                xshift=5,
                                yshift=5,
                                opacity=0.8
                            )

                    fig_calc.update_layout(
                        hovermode='closest',
                        plot_bgcolor='white',
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGray',
                            title_font=dict(size=24, color='black'),
                            tickfont=dict(size=18, color='black'),
                            linecolor='black',
                            linewidth=2
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGray',
                            title_font=dict(size=24, color='black'),
                            tickfont=dict(size=18, color='black'),
                            linecolor='black',
                            linewidth=2
                        ),
                        hoverlabel=dict(
                            font_size=18,
                            font_family="Arial"
                        ),
                        coloraxis_colorbar=dict(
                            title_font=dict(size=20, color='black'),
                            tickfont=dict(size=16, color='black'),
                            thickness=20,
                            len=0.7
                        )
                    )

                    st.plotly_chart(fig_calc)

                st.markdown("---")
                st.subheader("Data Preview")
                st.dataframe(df_viz.head(10))

                st.subheader("Export Options")

                export_col1, export_col2 = st.columns(2)

                with export_col1:
                    csv_export = df_viz.to_csv(index=False)
                    st.download_button(
                        label="Download t-SNE Coordinates (CSV)",
                        data=csv_export,
                        file_name="tsne_computed_coordinates.csv",
                        mime="text/csv"
                    )

                with export_col2:
                    html_export = fig_calc.to_html()
                    st.download_button(
                        label="Download Interactive Plot (HTML)",
                        data=html_export,
                        file_name="tsne_computed_plot.html",
                        mime="text/html"
                    )

    with fingerprint_tabs[0]:
        st.subheader("Principal Component Analysis")

        fingerprint_file = st.file_uploader("Upload fingerprints file (.npy or .npz)",
                                            type=['npy', 'npz'],
                                            key='pca_fingerprints')

        if fingerprint_file is not None:
            energies = None
            structure_names = None

            if fingerprint_file.name.endswith('.npz'):
                data = np.load(fingerprint_file)
                fingerprints = data['fingerprints']
                structure_names = data['structure_names'].tolist() if 'structure_names' in data else None

                if 'energies' in data:
                    energies = data['energies']
                    valid_energy_count = np.sum(~np.isnan(energies))
                    if valid_energy_count > 0:
                        st.success(
                            f"✅ Energies loaded from .npz file: {valid_energy_count}/{len(energies)} structures with energy data")
                    else:
                        st.info("📋 No valid energy data found in .npz file")
                        energies = None
            else:
                fingerprints = np.load(fingerprint_file)

            st.success(f"✅ Loaded fingerprints: {fingerprints.shape[0]} structures, {fingerprints.shape[1]} features")

            if structure_names is not None:
                st.success(f"✅ Structure names loaded from .npz file: {len(structure_names)} structures")


            coord_file = st.file_uploader(
                "Upload CSV/TXT with energies (optional - names loaded from .npz if available)",
                type=['csv', 'txt', 'dat'],
                key='pca_csv',
                help="Supports comma, tab, semicolon, space, or pipe separated files"
            )

            if coord_file is not None:
                try:
                    coord_file.seek(0)
                    first_line = coord_file.readline().decode('utf-8').strip()
                    coord_file.seek(0)

                    has_header = first_line.startswith('#')

                    if has_header:
                        st.info("📋 Header detected (line starting with #)")
                    else:
                        st.info("📋 No header detected - treating first line as data")

                    separators = [',', '\t', ';', r'\s+', '|']
                    separator_names = ['comma', 'tab', 'semicolon', 'whitespace', 'pipe']

                    coord_df = None
                    detected_sep = None

                    for sep, name in zip(separators, separator_names):
                        try:
                            coord_file.seek(0)  # Reset file pointer

                            if has_header:
                                df_temp = pd.read_csv(coord_file, sep=sep, engine='python', comment='#')
                            else:
                                df_temp = pd.read_csv(coord_file, sep=sep, engine='python', header=None)

                            if len(df_temp.columns) == 2:
                                coord_df = df_temp
                                detected_sep = name
                                break
                            elif len(df_temp.columns) > 1 and coord_df is None:
                                coord_df = df_temp
                                detected_sep = name
                        except:
                            continue

                    if coord_df is None:
                        raise ValueError("Could not parse file with any common separator")

                    st.success(f"✅ File parsed successfully using **{detected_sep}** separator")
                    st.info(f"📄 Loaded: {len(coord_df)} rows, columns: {list(coord_df.columns)}")

                    if len(coord_df.columns) == 2:
                        coord_df.columns = ['structure', 'energy']
                        st.info("🔧 Assigned column names: 'structure' and 'energy'")
                    elif 'structure' not in coord_df.columns or 'energy' not in coord_df.columns:
                        st.warning("⚠️ Could not automatically assign column names")


                    if 'structure' in coord_df.columns:
                        coord_df['structure'] = coord_df['structure'].astype(str).str.strip()

                    if 'energy' in coord_df.columns:
                        coord_df['energy'] = pd.to_numeric(coord_df['energy'], errors='coerce')

                    if structure_names is None and 'structure' in coord_df.columns:
                        structure_names = coord_df['structure'].tolist()
                        st.success(f"✅ Structure names loaded: {len(structure_names)} structures")

                    if structure_names is not None:
                        structure_names = [str(name).strip() for name in structure_names]

                    if 'energy' in coord_df.columns:
                        if 'structure' in coord_df.columns and structure_names is not None:
                            energy_dict = dict(zip(coord_df['structure'], coord_df['energy']))
                            energies = np.array([energy_dict.get(name, np.nan) for name in structure_names])

                            matched_count = np.sum(~np.isnan(energies))

                            if matched_count == len(structure_names):
                                st.success(f"✅ Perfect match! All {matched_count} structures have energy data")
                            else:
                                st.success(
                                    f"✅ Energies loaded: {matched_count}/{len(structure_names)} structures matched")

                            if matched_count < len(structure_names):
                                unmatched = [name for name, e in zip(structure_names, energies) if np.isnan(e)]
                                st.warning(f"⚠️ {len(unmatched)} structures without energy data")


                                with st.expander("🔍 Debug: Show mismatches and all structures"):
                                    st.write(f"**Total structures in .npz:** {len(structure_names)}")
                                    st.write(f"**Total structures in energy file:** {len(energy_dict)}")
                                    st.write("")

                                    st.write("**Unmatched structures from .npz:**")
                                    for name in unmatched:
                                        st.write(f"  - `{repr(name)}` (length: {len(name)})")

                                    st.write("")
                                    st.write("**All structures in .npz file:**")
                                    for i, name in enumerate(structure_names[:10]):
                                        matched_status = "✓" if name in energy_dict else "✗"
                                        st.write(f"  {matched_status} `{name}`")
                                    if len(structure_names) > 10:
                                        st.write(f"  ... and {len(structure_names) - 10} more")

                                    st.write("")
                                    st.write("**All structures in energy file:**")
                                    for i, name in enumerate(list(energy_dict.keys())[:10]):
                                        st.write(f"  - `{name}`")
                                    if len(energy_dict) > 10:
                                        st.write(f"  ... and {len(energy_dict) - 10} more")
                        else:
                            energies = coord_df['energy'].values
                            st.success(f"✅ Energies loaded: {len(energies)} values (order-based matching)")
                    else:
                        st.warning(f"⚠️ No 'energy' column found. Available columns: {list(coord_df.columns)}")
                        st.info("💡 Expected format: two columns with structure names and energy values")

                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    st.info("💡 **Expected file format:**")
                    st.code("""# Optional header line starting with #
            config_1.poscar	-4.563
            config_2.poscar	-4.621

            Or with comma/space/semicolon:
            config_1.poscar,-4.563
            config_2.poscar,-4.621""")

            pca_tabs = st.tabs([
                "📈 PCA, k-Means Clustering",
                "⚙️ PCA Settings & Computation",
                "📊 Variance Analysis",
                "🗺️ PCA Scatter Plots",
                "📈 Summary & Export",
                "🔧 Generate Analysis Script"
            ])
            with pca_tabs[0]:
                st.subheader("PCA Dimensionality Reduction → 2D Visualization")

                st.info("Use PCA to reduce fingerprints to fewer dimensions, then visualize with t-SNE or UMAP")

                reduction_col1, reduction_col2 = st.columns([1, 2])

                with reduction_col1:
                    st.markdown("**Step 1: PCA Reduction**")

                    max_pca_components = min(5000, fingerprints.shape[0], fingerprints.shape[1])

                    default_pca_components = min(max_pca_components, max(10, int(max_pca_components * 0.99)))

                    n_pca_components = st.slider(
                        "Number of PCA components",
                        min_value=2,
                        max_value=max_pca_components,
                        value=default_pca_components,
                        step=1,
                        key='n_pca_reduction',
                        help=f"Maximum available: {max_pca_components} components (limited by min(n_samples, n_features))"
                    )

                    st.caption(
                        f"Data shape: {fingerprints.shape[0]} samples × {fingerprints.shape[1]} features → Max PCA components: {max_pca_components}")

                    apply_standardization_reduction = st.checkbox(
                        "Apply standardization",
                        value=False,
                        key='std_reduction'
                    )

                    st.markdown("---")
                    st.markdown("**Step 2: 2D Visualization Method**")

                    viz_methods = ["t-SNE"]
                    if UMAP_AVAILABLE:
                        viz_methods.append("UMAP")
                    else:
                        st.caption("⚠️ UMAP not available. Install with: pip install umap-learn")

                    viz_method = st.selectbox(
                        "Select method",
                        options=viz_methods,
                        index=1,
                        key='viz_method_selector'
                    )

                    if viz_method == "t-SNE":
                        st.markdown("**t-SNE Parameters**")
                        tsne_perp_red = st.number_input(
                            "Perplexity",
                            min_value=5,
                            max_value=50,
                            value=30,
                            step=5,
                            key='tsne_perp_red'
                        )
                        tsne_iter_red = st.number_input(
                            "Max iterations",
                            min_value=250,
                            max_value=5000,
                            value=1000,
                            step=250,
                            key='tsne_iter_red'
                        )
                        tsne_lr_red = st.number_input(
                            "Learning rate",
                            min_value=10.0,
                            max_value=1000.0,
                            value=200.0,
                            step=50.0,
                            key='tsne_lr_red'
                        )
                    else:
                        st.markdown("**UMAP Parameters**")
                        umap_neighbors = st.number_input(
                            "N neighbors",
                            min_value=2,
                            max_value=200,
                            value=15,
                            step=5,
                            key='umap_neighbors'
                        )
                        umap_min_dist = st.number_input(
                            "Min distance",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.1,
                            step=0.05,
                            key='umap_min_dist'
                        )

                    st.markdown("---")
                    st.markdown("**Step 3: Clustering (Optional)**")

                    perform_clustering_reduction = st.checkbox(
                        "Perform k-means clustering",
                        value=True,
                        key='cluster_reduction'
                    )

                    if perform_clustering_reduction:
                        n_clusters_reduction = st.slider(
                            "Number of clusters",
                            min_value=2,
                            max_value=20,
                            value=3,
                            step=1,
                            key='n_clusters_red'
                        )

                        cluster_on = st.radio(
                            "Cluster based on:",
                            ["PCA-reduced features", "2D coordinates"],
                            index=0,
                            key='cluster_on_red'
                        )

                        highlight_centroids = st.checkbox(
                            "Highlight closest to centroids",
                            value=False,
                            key='highlight_centroids'
                        )

                    st.markdown("---")

                    compute_reduction = st.button(
                        "Compute Reduction & Visualization",
                        type="primary",
                        key='compute_reduction_btn'
                    )

                with reduction_col2:
                    if compute_reduction or 'reduction_computed' in st.session_state:

                        if compute_reduction:
                            current_energies = energies if energies is not None else None
                            current_structure_names = structure_names if structure_names is not None else None
                            with st.spinner("Computing PCA reduction..."):
                                if apply_standardization_reduction:
                                    scaler_red = StandardScaler()
                                    fingerprints_scaled_red = scaler_red.fit_transform(fingerprints)
                                else:
                                    fingerprints_scaled_red = fingerprints

                                pca_reducer = PCA(n_components=n_pca_components)
                                pca_reduced = pca_reducer.fit_transform(fingerprints_scaled_red)

                                explained_var_reduced = pca_reducer.explained_variance_ratio_ * 100
                                cumulative_var_reduced = np.cumsum(explained_var_reduced)

                                st.success(f"✅ PCA reduced to {n_pca_components} components")
                                st.info(f"Cumulative variance explained: {cumulative_var_reduced[-1]:.2f}%")

                            with st.spinner(f"Computing {viz_method}..."):
                                if viz_method == "t-SNE":
                                    n_samples = pca_reduced.shape[0]
                                    actual_perp = min(tsne_perp_red, n_samples - 1) if n_samples > 1 else 1

                                    tsne_model = TSNE(
                                        n_components=2,
                                        perplexity=actual_perp,
                                        max_iter=int(tsne_iter_red),
                                        learning_rate=tsne_lr_red,
                                        random_state=42
                                    )
                                    coords_2d = tsne_model.fit_transform(pca_reduced)
                                else:
                                    umap_model = umap.UMAP(
                                        n_components=2,
                                        n_neighbors=int(umap_neighbors),
                                        min_dist=umap_min_dist,
                                        random_state=42
                                    )
                                    coords_2d = umap_model.fit_transform(pca_reduced)

                                st.success(f"✅ {viz_method} computation completed")

                            if perform_clustering_reduction:
                                with st.spinner("Performing k-means clustering..."):
                                    if cluster_on == "PCA-reduced features":
                                        cluster_data = pca_reduced
                                    else:
                                        cluster_data = coords_2d

                                    kmeans_red = KMeans(
                                        n_clusters=n_clusters_reduction,
                                        random_state=42,
                                        n_init=10
                                    )
                                    clusters_red = kmeans_red.fit_predict(cluster_data)
                                    centroids = kmeans_red.cluster_centers_

                                    if highlight_centroids:
                                        closest_to_centroids = {}
                                        for cluster_id in range(n_clusters_reduction):
                                            cluster_mask = clusters_red == cluster_id
                                            cluster_points = cluster_data[cluster_mask]
                                            cluster_indices = np.where(cluster_mask)[0]

                                            distances = np.linalg.norm(
                                                cluster_points - centroids[cluster_id],
                                                axis=1
                                            )
                                            closest_idx_in_cluster = np.argmin(distances)
                                            closest_idx_global = cluster_indices[closest_idx_in_cluster]

                                            closest_to_centroids[cluster_id] = closest_idx_global
                                    else:
                                        closest_to_centroids = {}

                                    st.success(f"✅ Clustering completed: {n_clusters_reduction} clusters")
                            else:
                                clusters_red = None
                                closest_to_centroids = {}

                            st.session_state['reduction_computed'] = True
                            st.session_state['pca_reduced'] = pca_reduced
                            st.session_state['coords_2d'] = coords_2d
                            st.session_state['clusters_red'] = clusters_red
                            st.session_state['closest_to_centroids'] = closest_to_centroids
                            st.session_state['viz_method_used'] = viz_method
                            st.session_state['n_pca_components'] = n_pca_components
                            st.session_state['explained_var_reduced'] = explained_var_reduced
                            st.session_state['cumulative_var_reduced'] = cumulative_var_reduced

                            st.session_state['energies_for_plot'] = current_energies
                            st.session_state['structure_names_for_plot'] = current_structure_names

                        pca_reduced = st.session_state['pca_reduced']
                        coords_2d = st.session_state['coords_2d']
                        clusters_red = st.session_state['clusters_red']
                        closest_to_centroids = st.session_state['closest_to_centroids']
                        viz_method_used = st.session_state['viz_method_used']

                        energies_plot = st.session_state.get('energies_for_plot', energies)
                        structure_names_plot = st.session_state.get('structure_names_for_plot', structure_names)

                        # Use energies from session state if available, otherwise use outer scope
                        energies_to_use = st.session_state.get('energies_for_plot', energies)
                        structure_names_to_use = st.session_state.get('structure_names_for_plot', structure_names)

                        if structure_names_to_use is not None and len(structure_names_to_use) == len(coords_2d):
                            struct_names_plot = structure_names_to_use
                        else:
                            struct_names_plot = [f'Structure_{i + 1}' for i in range(len(coords_2d))]

                        plot_df = pd.DataFrame({
                            'x': coords_2d[:, 0],
                            'y': coords_2d[:, 1],
                            'structure': struct_names_plot
                        })

                        # Add energies if available - with debug info
                        if energies_to_use is not None and len(energies_to_use) == len(coords_2d):
                            plot_df['energy'] = energies_to_use
                            valid_count = plot_df['energy'].notna().sum()
                            st.success(f"✅ Energy data added to visualization: {valid_count} structures")
                        else:
                            if energies_to_use is not None:
                                st.warning(
                                    f"⚠️ Energy length mismatch: {len(energies_to_use)} energies vs {len(coords_2d)} coordinates")

                        if clusters_red is not None:
                            plot_df['cluster'] = clusters_red
                            plot_df['cluster_label'] = [f'Cluster {c + 1}' for c in clusters_red]

                        st.markdown(f"### {viz_method_used} Visualization from PCA-Reduced Data")

                        # Plotting controls
                        st.markdown("**Plot Settings**")

                        point_size_red = st.slider(
                            "Point size",
                            min_value=5,
                            max_value=50,
                            value=20,
                            step=5,
                            key='point_size_red'
                        )

                        show_labels_red = st.checkbox(
                            "Show structure labels",
                            value=False,
                            key='show_labels_red'
                        )

                        st.markdown("---")
                        st.markdown("**Coloring Options**")

                        coloring_options = ["Default (single color)"]

                        if clusters_red is not None:
                            coloring_options.append("By Cluster")

                        if 'energy' in plot_df.columns and plot_df['energy'].notna().any():
                            coloring_options.append("By Energy")

                        color_mode = st.radio(
                            "Color points by:",
                            options=coloring_options,
                            index=1 if len(coloring_options) > 1 else 0,
                            key='color_mode_red'
                        )

                        if color_mode == "By Energy":
                            cmap_red = st.selectbox(
                                "Colormap",
                                ["Viridis", "Plasma", "Inferno", "Magma", "Turbo", "RdYlBu_r"],
                                index=0,
                                key='cmap_red'
                            )

                            st.markdown("**Color Scale Range**")
                            energy_min_val = float(plot_df['energy'].min())
                            energy_max_val = float(plot_df['energy'].max())

                            use_custom_range_red = st.checkbox("Use custom range", value=False, key='custom_range_red')

                            if use_custom_range_red:
                                color_min_red = st.number_input("Minimum", value=energy_min_val, step=0.001,
                                                                format="%.6f", key='color_min_red')
                                color_max_red = st.number_input("Maximum", value=energy_max_val, step=0.001,
                                                                format="%.6f", key='color_max_red')
                            else:
                                color_min_red = energy_min_val
                                color_max_red = energy_max_val

                        st.markdown("---")

                        if color_mode == "By Cluster":
                            fig_red = px.scatter(
                                plot_df,
                                x='x',
                                y='y',
                                color='cluster_label',
                                hover_data=['structure', 'energy', 'cluster'] if 'energy' in plot_df.columns else [
                                    'structure', 'cluster'],
                                labels={
                                    'x': f'{viz_method_used} Component 1',
                                    'y': f'{viz_method_used} Component 2',
                                    'cluster_label': 'Cluster'
                                },
                                title=f'{viz_method_used} from {st.session_state["n_pca_components"]} PCA Components - Clustered',
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                width=900,
                                height=700
                            )

                            fig_red.update_traces(
                                marker=dict(size=point_size_red, line=dict(width=1, color='DarkSlateGrey'))
                            )

                            if len(closest_to_centroids) > 0:
                                for cluster_id, closest_idx in closest_to_centroids.items():
                                    fig_red.add_trace(
                                        go.Scatter(
                                            x=[coords_2d[closest_idx, 0]],
                                            y=[coords_2d[closest_idx, 1]],
                                            mode='markers',
                                            marker=dict(
                                                size=point_size_red * 2,
                                                color='gold',
                                                symbol='star',
                                                line=dict(width=3, color='black')
                                            ),
                                            name=f'Centroid {cluster_id + 1}',
                                            showlegend=True,
                                            hovertext=f'Closest to Cluster {cluster_id + 1} centroid<br>{struct_names_plot[closest_idx]}',
                                            hoverinfo='text'
                                        )
                                    )

                        elif color_mode == "By Energy":
                            fig_red = px.scatter(
                                plot_df,
                                x='x',
                                y='y',
                                color='energy',
                                hover_data=['structure', 'energy'],
                                labels={
                                    'x': f'{viz_method_used} Component 1',
                                    'y': f'{viz_method_used} Component 2',
                                    'energy': 'Energy (eV)'
                                },
                                title=f'{viz_method_used} from {st.session_state["n_pca_components"]} PCA Components - Colored by Energy',
                                color_continuous_scale=cmap_red.lower(),
                                range_color=[color_min_red, color_max_red],
                                width=900,
                                height=700
                            )

                            fig_red.update_traces(
                                marker=dict(size=point_size_red, line=dict(width=1, color='DarkSlateGrey'))
                            )

                        else:
                            fig_red = px.scatter(
                                plot_df,
                                x='x',
                                y='y',
                                hover_data=['structure', 'energy'] if 'energy' in plot_df.columns else ['structure'],
                                labels={
                                    'x': f'{viz_method_used} Component 1',
                                    'y': f'{viz_method_used} Component 2'
                                },
                                title=f'{viz_method_used} from {st.session_state["n_pca_components"]} PCA Components',
                                width=900,
                                height=700
                            )

                            fig_red.update_traces(
                                marker=dict(size=point_size_red, color='steelblue',
                                            line=dict(width=1, color='DarkSlateGrey'))
                            )

                        if show_labels_red:
                            for i, row in plot_df.iterrows():
                                fig_red.add_annotation(
                                    x=row['x'],
                                    y=row['y'],
                                    text=row['structure'],
                                    showarrow=False,
                                    font=dict(size=10),
                                    xshift=5,
                                    yshift=5,
                                    opacity=0.7
                                )

                        fig_red.update_layout(
                            hovermode='closest',
                            plot_bgcolor='white',
                            xaxis=dict(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='LightGray',
                                title_font=dict(size=20, color='black'),
                                tickfont=dict(size=16, color='black'),
                                linecolor='black',
                                linewidth=2
                            ),
                            yaxis=dict(
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='LightGray',
                                title_font=dict(size=20, color='black'),
                                tickfont=dict(size=16, color='black'),
                                linecolor='black',
                                linewidth=2
                            ),
                            hoverlabel=dict(
                                font_size=16,
                                font_family="Arial"
                            ),
                            legend=dict(
                                font=dict(size=14),
                                bgcolor='rgba(255,255,255,0.8)'
                            )
                        )

                        st.plotly_chart(fig_red,)

                        if len(closest_to_centroids) > 0:
                            st.markdown("---")
                            st.subheader("🎯 Structures Closest to Cluster Centroids")

                            centroid_data = []
                            for cluster_id, closest_idx in sorted(closest_to_centroids.items()):
                                struct_name = struct_names_plot[closest_idx]
                                energy_val = energies[closest_idx] if energies is not None else None

                                centroid_data.append({
                                    'Cluster': f'Cluster {cluster_id + 1}',
                                    'Structure': struct_name,
                                    'Energy (eV)': f'{energy_val:.6f}' if energy_val is not None else 'N/A',
                                    'X': f'{coords_2d[closest_idx, 0]:.4f}',
                                    'Y': f'{coords_2d[closest_idx, 1]:.4f}'
                                })

                            centroid_df = pd.DataFrame(centroid_data)

                            st.dataframe(
                                centroid_df,
                                hide_index=True
                            )

                            st.markdown("**Export Centroid Structures**")

                            csv_centroids = centroid_df.to_csv(index=False)
                            st.download_button(
                                label="Download Centroid Structures (CSV)",
                                data=csv_centroids,
                                file_name=f"centroid_structures_{viz_method_used}.csv",
                                mime="text/csv",
                                type='primary',
                                key='download_centroids'
                            )

                        st.markdown("---")
                        st.subheader("📥 Export All Data")
                        export_cols = st.columns(4)

                        with export_cols[0]:
                            full_export_df = plot_df.copy()
                            csv_full = full_export_df.to_csv(index=False)
                            st.download_button(
                                label=f"Download {viz_method_used} Coordinates (CSV)",
                                data=csv_full,
                                file_name=f"pca_reduced_{viz_method_used.lower()}_coordinates.csv",
                                mime="text/csv",
                                type='primary',
                                key='download_coords'
                            )

                        with export_cols[1]:
                            html_export_red = fig_red.to_html()
                            st.download_button(
                                label="Download Interactive Plot (HTML)",
                                data=html_export_red,
                                file_name=f"pca_reduced_{viz_method_used.lower()}_plot.html",
                                mime="text/html",
                                type='primary',
                                key='download_html'
                            )

                        with export_cols[2]:
                            if clusters_red is not None:
                                cluster_assignment_df = pd.DataFrame({
                                    'structure': struct_names_plot,
                                    'cluster': clusters_red + 1
                                })

                                if energies is not None and len(energies) == len(struct_names_plot):
                                    cluster_assignment_df['energy'] = energies

                                cluster_assignment_df['x'] = coords_2d[:, 0]
                                cluster_assignment_df['y'] = coords_2d[:, 1]

                                if 'energy' in cluster_assignment_df.columns:
                                    cluster_assignment_df = cluster_assignment_df.sort_values(
                                        by=['cluster', 'energy'],
                                        ascending=[True, True],
                                        na_position='last'
                                    ).reset_index(drop=True)
                                else:
                                    cluster_assignment_df = cluster_assignment_df.sort_values(
                                        by='cluster',
                                        ascending=True
                                    ).reset_index(drop=True)

                                csv_cluster_assignments = cluster_assignment_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Cluster Assignments (CSV)",
                                    data=csv_cluster_assignments,
                                    file_name=f"cluster_assignments_{viz_method_used.lower()}.csv",
                                    mime="text/csv",
                                    type='primary',
                                    key='download_cluster_assignments'
                                )
                            else:
                                st.info("Enable clustering to export assignments")

                        with export_cols[3]:
                            if clusters_red is not None:
                                cluster_summary = []
                                for cluster_id in range(len(set(clusters_red))):
                                    cluster_mask = clusters_red == cluster_id
                                    cluster_structures = [struct_names_plot[i] for i in np.where(cluster_mask)[0]]
                                    cluster_energies = energies[cluster_mask] if energies is not None else None

                                    summary_entry = {
                                        'Cluster': f'Cluster {cluster_id + 1}',
                                        'N_Structures': int(cluster_mask.sum()),
                                        'Structures': '; '.join(cluster_structures)
                                    }

                                    if cluster_energies is not None:
                                        summary_entry['Mean_Energy'] = float(np.mean(cluster_energies))
                                        summary_entry['Min_Energy'] = float(np.min(cluster_energies))
                                        summary_entry['Max_Energy'] = float(np.max(cluster_energies))

                                    cluster_summary.append(summary_entry)

                                cluster_summary_df = pd.DataFrame(cluster_summary)
                                csv_cluster_summary = cluster_summary_df.to_csv(index=False)

                                st.download_button(
                                    label="Download Cluster Summary (CSV)",
                                    data=csv_cluster_summary,
                                    file_name=f"cluster_summary_{viz_method_used.lower()}.csv",
                                    mime="text/csv",
                                    type='primary',
                                    key='download_cluster_summary'
                                )
                    else:
                        st.info("👈 Configure parameters and click 'Compute Reduction & Visualization' to start")
            with pca_tabs[1]:
                st.subheader("PCA Configuration")

                col1, col2 = st.columns(2)

                with col1:
                    apply_standardization = st.checkbox("Apply standardization", value=True)
                    st.caption("Recommended: scales features to have mean=0 and std=1")

                    n_components = st.slider(
                        "Number of components to compute",
                        min_value=2,
                        max_value=20,
                        value=10,
                        step=1
                    )
                    st.caption("Total components to analyze")

                with col2:
                    variance_threshold = st.slider(
                        "Cumulative variance threshold (%)",
                        min_value=80,
                        max_value=99,
                        value=95,
                        step=1
                    )
                    st.caption("Highlight components explaining this % of variance")

                max_components = min(fingerprints.shape[0], fingerprints.shape[1])
                if n_components > max_components:
                    n_components = max_components
                    st.warning(f"⚠️ Adjusted n_components to {max_components} (limited by data dimensions)")

                if apply_standardization:
                    scaler = StandardScaler()
                    fingerprints_scaled = scaler.fit_transform(fingerprints)
                    st.info("🔧 Standardization applied")
                else:
                    fingerprints_scaled = fingerprints

                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(fingerprints_scaled)

                explained_variance = pca.explained_variance_ratio_ * 100
                cumulative_variance = np.cumsum(explained_variance)

                st.success("✅ PCA computation completed successfully!")

            with pca_tabs[2]:
                st.subheader("Variance Analysis")

                st.markdown("#### Variance Explained by Each Component")

                fig_var = go.Figure()

                fig_var.add_trace(go.Bar(
                    x=np.arange(1, len(explained_variance) + 1),
                    y=explained_variance,
                    name='Individual',
                    marker_color='steelblue'
                ))

                fig_var.update_layout(
                    xaxis_title="Principal Component",
                    yaxis_title="Explained Variance (%)",
                    showlegend=True,
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGray',
                        dtick=1,
                        title_font=dict(size=24, color='black'),
                        tickfont=dict(size=18, color='black'),
                        linecolor='black',
                        linewidth=2
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGray',
                        title_font=dict(size=24, color='black'),
                        tickfont=dict(size=18, color='black'),
                        linecolor='black',
                        linewidth=2
                    )
                )

                st.plotly_chart(fig_var, )

                st.markdown("---")

                st.markdown("#### Cumulative Explained Variance")

                fig_cum = go.Figure()

                fig_cum.add_trace(go.Scatter(
                    x=np.arange(1, len(cumulative_variance) + 1),
                    y=cumulative_variance,
                    mode='lines+markers',
                    name='Cumulative Variance',
                    line=dict(color='darkgreen', width=3),
                    marker=dict(size=8)
                ))

                fig_cum.add_hline(
                    y=variance_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{variance_threshold}% threshold",
                    annotation_position="right",
                    annotation_font=dict(size=16, color='red')
                )

                n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1

                fig_cum.add_vline(
                    x=n_components_threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"PC{n_components_threshold}",
                    annotation_position="top",
                    annotation_font=dict(size=16, color='orange')
                )

                fig_cum.update_layout(
                    xaxis_title="Principal Component",
                    yaxis_title="Cumulative Explained Variance (%)",
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGray',
                        dtick=1,
                        title_font=dict(size=24, color='black'),
                        tickfont=dict(size=18, color='black'),
                        linecolor='black',
                        linewidth=2
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='LightGray',
                        range=[0, 105],
                        title_font=dict(size=24, color='black'),
                        tickfont=dict(size=18, color='black'),
                        linecolor='black',
                        linewidth=2
                    )
                )

                st.plotly_chart(fig_cum, )

                st.success(f"✨ **{n_components_threshold} components** explain **{variance_threshold}%** of the variance")

            with pca_tabs[3]:
                st.subheader("PCA Scatter Plots")

                scatter_col1, scatter_col2 = st.columns([1, 2])

                with scatter_col1:
                    st.markdown("**Plot Settings**")

                    pc_x = st.selectbox("X-axis", options=list(range(1, n_components + 1)), index=0, key='pc_x')
                    pc_y = st.selectbox("Y-axis", options=list(range(1, n_components + 1)), index=1, key='pc_y')

                    pca_point_size = st.slider("Point size", min_value=5, max_value=50, value=25, step=5,
                                               key='pca_point')
                    show_pca_labels = st.checkbox("Show structure labels", value=False, key='pca_labels')

                    if energies is not None:
                        color_by_energy_pca = st.checkbox("Color by energy", value=True, key='pca_energy')
                        if color_by_energy_pca:
                            pca_colormap = st.selectbox(
                                "Colormap",
                                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", "RdYlBu_r"],
                                index=0,
                                key='pca_cmap'
                            )
                    else:
                        color_by_energy_pca = False

                with scatter_col2:
                    pca_df = pd.DataFrame({
                        f'PC{i + 1}': pca_result[:, i] for i in range(n_components)
                    })

                    if structure_names is not None:
                        pca_df['structure'] = structure_names
                    else:
                        pca_df['structure'] = [f'Structure {i + 1}' for i in range(len(pca_result))]

                    if energies is not None:
                        pca_df['energy'] = energies

                    if energies is not None and color_by_energy_pca:
                        fig_pca = px.scatter(
                            pca_df,
                            x=f'PC{pc_x}',
                            y=f'PC{pc_y}',
                            color='energy',
                            hover_data=['structure', 'energy'],
                            labels={
                                f'PC{pc_x}': f'PC{pc_x} ({explained_variance[pc_x - 1]:.2f}%)',
                                f'PC{pc_y}': f'PC{pc_y} ({explained_variance[pc_y - 1]:.2f}%)',
                                'energy': 'Energy (eV)'
                            },
                            title=f'PCA: PC{pc_x} vs PC{pc_y}',
                            color_continuous_scale=pca_colormap.lower(),
                            width=800,
                            height=600
                        )
                    else:
                        fig_pca = px.scatter(
                            pca_df,
                            x=f'PC{pc_x}',
                            y=f'PC{pc_y}',
                            hover_data=['structure'],
                            labels={
                                f'PC{pc_x}': f'PC{pc_x} ({explained_variance[pc_x - 1]:.2f}%)',
                                f'PC{pc_y}': f'PC{pc_y} ({explained_variance[pc_y - 1]:.2f}%)'
                            },
                            title=f'PCA: PC{pc_x} vs PC{pc_y}',
                            width=800,
                            height=600
                        )

                    fig_pca.update_traces(marker=dict(size=pca_point_size, line=dict(width=1, color='DarkSlateGrey')))

                    if show_pca_labels:
                        for i, row in pca_df.iterrows():
                            fig_pca.add_annotation(
                                x=row[f'PC{pc_x}'],
                                y=row[f'PC{pc_y}'],
                                text=row['structure'],
                                showarrow=False,
                                font=dict(size=12),
                                xshift=5,
                                yshift=5,
                                opacity=0.8
                            )

                    fig_pca.update_layout(
                        hovermode='closest',
                        plot_bgcolor='white',
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGray',
                            title_font=dict(size=24, color='black'),
                            tickfont=dict(size=18, color='black'),
                            linecolor='black',
                            linewidth=2
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='LightGray',
                            title_font=dict(size=24, color='black'),
                            tickfont=dict(size=18, color='black'),
                            linecolor='black',
                            linewidth=2
                        ),
                        hoverlabel=dict(
                            font_size=18,
                            font_family="Arial"
                        ),
                        coloraxis_colorbar=dict(
                            title_font=dict(size=20, color='black'),
                            tickfont=dict(size=16, color='black'),
                            thickness=20,
                            len=0.7
                        )
                    )

                    st.plotly_chart(fig_pca, )

            with pca_tabs[4]:
                st.subheader("Summary Statistics")

                summary_col1, summary_col2, summary_col3 = st.columns(3)

                with summary_col1:
                    st.metric("Total Components", n_components)
                    st.metric("Components for 95% variance", np.argmax(cumulative_variance >= 95) + 1)

                with summary_col2:
                    st.metric("PC1 Variance", f"{explained_variance[0]:.2f}%")
                    st.metric("PC1+PC2 Variance", f"{cumulative_variance[1]:.2f}%")

                with summary_col3:
                    st.metric(f"Components for {variance_threshold}% variance", n_components_threshold)
                    st.metric("Total structures", fingerprints.shape[0])

                st.markdown("---")
                st.subheader("Export PCA Results")

                export_col1, export_col2, export_col3 = st.columns(3)

                with export_col1:
                    csv_pca = pca_df.to_csv(index=False)
                    st.download_button(
                        label="Download PCA Coordinates (CSV)",
                        data=csv_pca,
                        file_name="pca_coordinates.csv",
                        mime="text/csv"
                    )

                with export_col2:
                    variance_df = pd.DataFrame({
                        'Component': [f'PC{i + 1}' for i in range(len(explained_variance))],
                        'Explained_Variance_%': explained_variance,
                        'Cumulative_Variance_%': cumulative_variance
                    })
                    csv_variance = variance_df.to_csv(index=False)
                    st.download_button(
                        label="Download Variance Data (CSV)",
                        data=csv_variance,
                        file_name="pca_variance.csv",
                        mime="text/csv"
                    )

                with export_col3:
                    html_pca = fig_pca.to_html()
                    st.download_button(
                        label="Download PCA Plot (HTML)",
                        data=html_pca,
                        file_name="pca_plot.html",
                        mime="text/html"
                    )

with tab_neighbor:
    st.header("Nearest Neighbor Distance Analysis")

    st.markdown("""
    Analyze nearest neighbor distances between element pairs in your structures.
    This analysis shows the distribution of atomic distances and can reveal structural patterns.
    """)

    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **This analysis calculates:**
        - Distance distributions between pairs of elements
        - Histogram of how many atom pairs exist at each distance
        - Energy correlation with structural features

        **Use this to:**
        - Identify characteristic bond lengths
        - Compare local structure across different compositions
        - Find correlations between structure and stability
        """)

    neighbor_tabs = st.tabs([
        "🔧 Generate Analysis Script",
        "📊 Visualize Results"
    ])

    with neighbor_tabs[0]:
        st.subheader("Generate Nearest Neighbor Analysis Script")

        st.info("This script will analyze all element pairs involving your chosen reference element")

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.markdown("**Analysis Parameters**")
            reference_element = st.text_input(
                "Reference Element",
                value="Ti",
                help="Element to use as reference for pair analysis"
            )

            include_second_neighbor = st.checkbox(
                "Include Second Nearest Neighbor",
                value=False,
                help="Calculate both first and second nearest neighbor distances"
            )

            max_distance = st.number_input(
                "Maximum Distance (Å)",
                min_value=2.0,
                max_value=15.0,
                value=8.0,
                step=0.5,
                help="Maximum distance to consider for neighbors"
            )

            num_bins = st.number_input(
                "Number of Distance Bins",
                min_value=20,
                max_value=200,
                value=80,
                step=10,
                help="Resolution of the distance histogram"
            )

        with config_col2:
            st.markdown("**File Settings**")

            structures_folder = st.text_input(
                "Structures Folder",
                value=".",
                help="Folder containing VASP files (use '.' for current directory)"
            )

            nn_energies_file = st.text_input(
                "Energy File Name",
                value="energies.txt",
                help="Text file with structure names and energies"
            )

            nn_output_file = st.text_input(
                "Output CSV File",
                value="nearest_neighbor_analysis.csv",
                help="Name for the output CSV file"
            )


        def generate_neighbor_script():
            script = f'''#!/usr/bin/env python3
"""
Nearest Neighbor Distance Analysis Script
Calculates the first (and optionally second) nearest neighbor distances for element pairs
"""

import numpy as np
from ase.io import read
from pathlib import Path
import pandas as pd

def read_energies(energies_file='{nn_energies_file}'):
    """Read energies from text file"""
    energies = {{}}
    try:
        with open(energies_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    energy = float(parts[1])
                    energies[filename] = energy
        print(f"Loaded energies for {{len(energies)}} structures")
    except FileNotFoundError:
        print(f"Warning: {{energies_file}} not found. Energies will be None.")
    except Exception as e:
        print(f"Error loading energies: {{str(e)}}")
    return energies

def get_nearest_neighbors(atoms, element1, element2, include_second={include_second_neighbor}):
    """
    Get the first (and optionally second) nearest neighbor distances between two elements
    Returns: list of (neighbor_order, distance, count) tuples
    """
    # Get positions of each element
    elem1_indices = [i for i, atom in enumerate(atoms) if atom.symbol == element1]
    elem2_indices = [i for i, atom in enumerate(atoms) if atom.symbol == element2]

    if not elem1_indices or not elem2_indices:
        return []

    positions1 = atoms.positions[elem1_indices]
    positions2 = atoms.positions[elem2_indices]

    # Handle periodic boundary conditions
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    all_distances = []

    # For each atom of element1, find distances to all element2 atoms
    for pos1 in positions1:
        if np.any(pbc):
            distances_for_this_atom = []
            for pos2 in positions2:
                delta = pos2 - pos1

                # Apply minimum image convention
                if pbc[0]:
                    delta[0] -= cell[0, 0] * np.round(delta[0] / cell[0, 0])
                if pbc[1]:
                    delta[1] -= cell[1, 1] * np.round(delta[1] / cell[1, 1])
                if pbc[2]:
                    delta[2] -= cell[2, 2] * np.round(delta[2] / cell[2, 2])

                dist = np.linalg.norm(delta)
                if dist > 0.1:  # Avoid self-interaction
                    distances_for_this_atom.append(dist)
        else:
            distances_for_this_atom = np.linalg.norm(positions2 - pos1, axis=1)
            distances_for_this_atom = distances_for_this_atom[distances_for_this_atom > 0.1]

        all_distances.extend(distances_for_this_atom)

    if len(all_distances) == 0:
        return []

    all_distances = sorted(all_distances)

    # Find first nearest neighbor distance
    tolerance = 0.1  # Angstrom

    first_nn = all_distances[0]
    first_nn_distances = [d for d in all_distances if abs(d - first_nn) < tolerance]
    count_first = len(first_nn_distances)

    results = [(1, first_nn, count_first)]

    # Find second nearest neighbor if requested
    if include_second:
        remaining = [d for d in all_distances if abs(d - first_nn) >= tolerance]
        if len(remaining) > 0:
            second_nn = remaining[0]
            second_nn_distances = [d for d in remaining if abs(d - second_nn) < tolerance]
            count_second = len(second_nn_distances)
            results.append((2, second_nn, count_second))

    return results

def get_unique_elements(structures_dir):
    """Get all unique elements from all structures"""
    elements = set()
    for file_path in Path(structures_dir).glob('*.vasp'):
        try:
            atoms = read(str(file_path))
            elements.update(atoms.get_chemical_symbols())
        except:
            continue
    return sorted(list(elements))

def main():
    print("="*60)
    print("Nearest Neighbors Analysis")
    print("="*60)
    print()

    STRUCTURES_DIR = "{structures_folder}"
    REFERENCE_ELEMENT = "{reference_element}"
    ENERGIES_FILE = "{nn_energies_file}"
    OUTPUT_FILE = "{nn_output_file}"
    INCLUDE_SECOND = {include_second_neighbor}

    print(f"Reference element: {{REFERENCE_ELEMENT}}")
    print(f"Include second neighbor: {{INCLUDE_SECOND}}")
    print()

    # Load energies
    energies = read_energies(ENERGIES_FILE)

    # Get all unique elements
    print("Scanning structures for elements...")
    all_elements = get_unique_elements(STRUCTURES_DIR)
    print(f"Found elements: {{all_elements}}")

    if REFERENCE_ELEMENT not in all_elements:
        print(f"ERROR: Reference element {{REFERENCE_ELEMENT}} not found!")
        return

    # Create element pairs
    element_pairs = [(REFERENCE_ELEMENT, elem) for elem in all_elements]
    print(f"Analyzing pairs: {{element_pairs}}")
    print()

    results = []
    structures = list(Path(STRUCTURES_DIR).glob('*.vasp'))
    total = len(structures)

    print(f"Processing {{total}} structures...")
    print()

    for idx, file_path in enumerate(structures, 1):
        filename = file_path.name
        print(f"[{{idx}}/{{total}}] {{filename}}")

        try:
            atoms = read(str(file_path))
            energy = energies.get(filename, None)

            composition = {{}}
            for atom in atoms:
                composition[atom.symbol] = composition.get(atom.symbol, 0) + 1

            for elem1, elem2 in element_pairs:
                neighbors = get_nearest_neighbors(atoms, elem1, elem2, INCLUDE_SECOND)

                for neighbor_order, distance, count in neighbors:
                    results.append({{
                        'filename': filename,
                        'element_pair': f"{{elem1}}-{{elem2}}",
                        'neighbor_order': neighbor_order,
                        'distance': distance,
                        'intensity': count,
                        'energy': energy,
                        'composition': str(composition)
                    }})

        except Exception as e:
            print(f"  Error: {{str(e)}}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {{OUTPUT_FILE}}")
    print(f"Total data points: {{len(df)}}")
    print(f"Element pairs: {{df['element_pair'].nunique()}}")
    print(f"Structures analyzed: {{df['filename'].nunique()}}")
    print()

    print("Summary by element pair:")
    for pair in sorted(df['element_pair'].unique()):
        pair_data = df[df['element_pair'] == pair]
        first_nn_data = pair_data[pair_data['neighbor_order'] == 1]
        second_nn_data = pair_data[pair_data['neighbor_order'] == 2]
        print(f"  {{pair}}:")
        if len(first_nn_data) > 0:
            print(f"    1st NN: {{first_nn_data['distance'].min():.2f}}-{{first_nn_data['distance'].max():.2f}} Å")
        if len(second_nn_data) > 0:
            print(f"    2nd NN: {{second_nn_data['distance'].min():.2f}}-{{second_nn_data['distance'].max():.2f}} Å")

if __name__ == "__main__":
    main()
'''
            return script


        if st.button("Generate Analysis Script", type="primary", key="gen_neighbor_script"):
            script_content = generate_neighbor_script()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_filename = f"nearest_neighbor_analysis_{timestamp}.py"

            st.success("Script generated successfully!")

            st.download_button(
                label="Download Python Script",
                data=script_content,
                file_name=script_filename,
                mime="text/x-python",
                type="primary"
            )

            with st.expander("View Generated Script", expanded=True):
                st.code(script_content, language="python")

    with neighbor_tabs[1]:
        st.subheader("Visualize Nearest Neighbor Analysis")

        st.info("Upload the CSV file generated by the analysis script")

        nn_csv_file = st.file_uploader(
            "Upload nearest_neighbor_analysis.csv",
            type=['csv'],
            key='nn_csv_upload'
        )

        if nn_csv_file is not None:
            df_nn = pd.read_csv(nn_csv_file)


            first_pair = df_nn['element_pair'].iloc[0]
            reference_elem = first_pair.split('-')[0]
            element_pairs = sorted(df_nn['element_pair'].unique())

            neighbor_orders = sorted(df_nn['neighbor_order'].unique())

            st.success(f"✅ Loaded data: {df_nn['filename'].nunique()} structures, {len(element_pairs)} element pairs")

            viz_col1, viz_col2 = st.columns([2, 1])

            with viz_col2:
                st.markdown("**Plot Settings**")

                if len(neighbor_orders) > 1:
                    selected_neighbor_order = st.selectbox(
                        "Neighbor Order",
                        options=neighbor_orders,
                        index=0,
                        help="Select which nearest neighbor to display"
                    )
                    df_nn_filtered = df_nn[df_nn['neighbor_order'] == selected_neighbor_order]
                else:
                    selected_neighbor_order = neighbor_orders[0]
                    df_nn_filtered = df_nn

                selected_pairs = st.multiselect(
                    "Element pairs to display",
                    options=element_pairs,
                    default=element_pairs[:min(6, len(element_pairs))],
                    help="Select which element pairs to show"
                )

                marker_size = st.slider("Marker size", 2, 12, 6, key='nn_marker')

                st.markdown("---")
                st.markdown("**Y-Axis Options**")

                y_axis_mode = st.radio(
                    "Y-axis variable:",
                    ["Intensity (# pairs)", "Energy (eV)"],
                    index=0,
                    help="Choose what to plot on the y-axis"
                )

                st.markdown("---")
                st.markdown("**Energy Filter**")

                if 'energy' in df_nn_filtered.columns and df_nn_filtered['energy'].notna().any():
                    energy_min_filter = float(df_nn_filtered['energy'].min())
                    energy_max_filter = float(df_nn_filtered['energy'].max())

                    enable_energy_filter = st.checkbox("Enable energy range filter", value=False)

                    if enable_energy_filter:
                        energy_range = st.slider(
                            "Select energy range (eV)",
                            min_value=energy_min_filter,
                            max_value=energy_max_filter,
                            value=(energy_min_filter, energy_max_filter),
                            step=(energy_max_filter - energy_min_filter) / 100,
                            format="%.4f"
                        )


                        df_nn_filtered = df_nn_filtered[
                            (df_nn_filtered['energy'] >= energy_range[0]) &
                            (df_nn_filtered['energy'] <= energy_range[1])
                            ]

                        st.info(f"Showing {df_nn_filtered['filename'].nunique()} structures in selected energy range")

                st.markdown("---")
                st.markdown("**Clustering & Coloring**")

                if 'energy' in df_nn_filtered.columns and df_nn_filtered['energy'].notna().any():
                    perform_clustering = st.checkbox("Perform k-means clustering on energies", value=False,
                                                     key='nn_cluster')

                    if perform_clustering:
                        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3,
                                               key='nn_n_clusters')

                        # Perform k-means clustering on energies
                        from sklearn.cluster import KMeans

                        energy_data = df_nn_filtered[['energy']].dropna()
                        valid_indices = df_nn_filtered['energy'].notna()

                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = np.full(len(df_nn_filtered), -1)
                        clusters[valid_indices] = kmeans.fit_predict(energy_data)
                        df_nn_filtered['cluster'] = clusters

                        color_by_cluster = st.checkbox("Color by cluster groups", value=True, key='nn_color_cluster')

                        if color_by_cluster:
                            st.info(f"Structures grouped into {n_clusters} energy-based clusters")
                    else:
                        color_by_cluster = False

                    if not perform_clustering or not color_by_cluster:
                        color_by_energy_nn = st.checkbox("Color by energy", value=True, key='nn_energy')

                        if color_by_energy_nn:
                            nn_colormap = st.selectbox(
                                "Colormap",
                                ["Viridis", "Plasma", "Inferno", "Magma", "RdYlBu_r", "Turbo"],
                                index=0,
                                key='nn_cmap'
                            )
                    else:
                        color_by_energy_nn = False
                else:
                    color_by_energy_nn = False
                    perform_clustering = False
                    color_by_cluster = False
                    st.info("No energy data found in file")

            with viz_col1:
                if selected_pairs:
                    n_pairs = len(selected_pairs)
                    n_cols = min(3, n_pairs)
                    n_rows = int(np.ceil(n_pairs / n_cols))

                    from plotly.subplots import make_subplots

                    if 'energy' in df_nn_filtered.columns and df_nn_filtered['energy'].notna().any():
                        global_min_energy = df_nn_filtered['energy'].min()
                        global_max_energy = df_nn_filtered['energy'].max()
                    else:
                        global_min_energy = None
                        global_max_energy = None

                    fig = make_subplots(
                        rows=n_rows,
                        cols=n_cols,
                        subplot_titles=selected_pairs,
                        vertical_spacing=0.12,
                        horizontal_spacing=0.08
                    )

                    for idx, pair in enumerate(selected_pairs):
                        row = idx // n_cols + 1
                        col = idx % n_cols + 1

                        pair_data = df_nn_filtered[df_nn_filtered['element_pair'] == pair]

                        if 'energy' in pair_data.columns and pair_data['energy'].notna().any():
                            min_energy_idx = pair_data['energy'].idxmin()
                            min_energy_structure = pair_data.loc[min_energy_idx, 'filename']
                            min_energy_value = pair_data['energy'].min()
                        else:
                            min_energy_structure = None
                            min_energy_value = None

                        all_x = []
                        all_y = []
                        all_colors = []
                        all_symbols = []
                        all_sizes = []
                        all_texts = []
                        all_cluster_labels = []

                        for filename in pair_data['filename'].unique():
                            structure_data = pair_data[pair_data['filename'] == filename].sort_values('distance')

                            energy = structure_data['energy'].iloc[0] if 'energy' in structure_data.columns else None
                            cluster_label = structure_data['cluster'].iloc[
                                0] if 'cluster' in structure_data.columns else None

                            is_min_energy = (filename == min_energy_structure)

                            for _, point in structure_data.iterrows():
                                all_x.append(point['distance'])
                                if y_axis_mode == "Energy (eV)":
                                    all_y.append(energy if energy is not None else np.nan)
                                else:
                                    all_y.append(point['intensity'])

                                if is_min_energy:
                                    all_colors.append(None)
                                    all_symbols.append('star')
                                    all_sizes.append(marker_size * 2)
                                else:
                                    if color_by_cluster:
                                        all_colors.append(cluster_label)
                                    else:
                                        if y_axis_mode == "Energy (eV)":
                                            all_colors.append(
                                                point['intensity'])
                                        else:
                                            all_colors.append(
                                                energy if energy is not None else None)
                                    all_symbols.append('circle')
                                    all_sizes.append(marker_size)

                                all_cluster_labels.append(cluster_label)

                                hover_text = f'<b>{filename}</b><br>'
                                hover_text += f'Distance: {point["distance"]:.3f} Å<br>'
                                hover_text += f'Intensity: {point["intensity"]}<br>'
                                if energy is not None:
                                    hover_text += f'Energy: {energy:.4f} eV<br>'
                                if y_axis_mode == "Energy (eV)":
                                    hover_text += f'<b>Y-value: {energy:.4f} eV</b><br>'
                                if color_by_cluster and cluster_label is not None and cluster_label >= 0:
                                    hover_text += f'Cluster: {int(cluster_label) + 1}<br>'
                                if is_min_energy:
                                    hover_text += '<b>⭐ MINIMUM ENERGY</b>'
                                all_texts.append(hover_text)

                        min_energy_mask = [s == 'star' for s in all_symbols]
                        if any(min_energy_mask):
                            fig.add_trace(
                                go.Scatter(
                                    x=[x for x, is_min in zip(all_x, min_energy_mask) if is_min],
                                    y=[y for y, is_min in zip(all_y, min_energy_mask) if is_min],
                                    mode='markers',
                                    marker=dict(
                                        size=[s for s, is_min in zip(all_sizes, min_energy_mask) if is_min],
                                        color='red',
                                        symbol='star',
                                        line=dict(width=2, color='darkred')
                                    ),
                                    showlegend=False,
                                    hovertemplate=[t for t, is_min in zip(all_texts, min_energy_mask) if is_min]
                                ),
                                row=row, col=col
                            )

                        regular_mask = [s == 'circle' for s in all_symbols]
                        if any(regular_mask):
                            regular_x = [x for x, is_reg in zip(all_x, regular_mask) if is_reg]
                            regular_y = [y for y, is_reg in zip(all_y, regular_mask) if is_reg]
                            regular_colors = [c for c, is_reg in zip(all_colors, regular_mask) if is_reg]
                            regular_sizes = [s for s, is_reg in zip(all_sizes, regular_mask) if is_reg]
                            regular_texts = [t for t, is_reg in zip(all_texts, regular_mask) if is_reg]

                            valid_colors = [c for c in regular_colors if c is not None]

                            if color_by_cluster and len(valid_colors) > 0:

                                import plotly.express as px

                                cluster_colors = px.colors.qualitative.Set3


                                cluster_numeric = []
                                for c in regular_colors:
                                    if c >= 0:
                                        cluster_numeric.append(int(c))
                                    else:
                                        cluster_numeric.append(-1)


                                point_colors = []
                                for c in cluster_numeric:
                                    if c >= 0:
                                        point_colors.append(cluster_colors[c % len(cluster_colors)])
                                    else:
                                        point_colors.append('gray')

                                cluster_names = [f'Cluster {int(c) + 1}' if c >= 0 else 'No data' for c in
                                                 regular_colors]

                                unique_clusters = sorted(list(set([c for c in cluster_numeric if c >= 0])))

                                for cluster_id in unique_clusters:
                                    cluster_mask = [c == cluster_id for c in cluster_numeric]
                                    if any(cluster_mask):
                                        fig.add_trace(
                                            go.Scatter(
                                                x=[x for x, m in zip(regular_x, cluster_mask) if m],
                                                y=[y for y, m in zip(regular_y, cluster_mask) if m],
                                                mode='markers',
                                                name=f'Cluster {cluster_id + 1}',
                                                marker=dict(
                                                    size=marker_size,
                                                    color=cluster_colors[cluster_id % len(cluster_colors)],
                                                    line=dict(width=0.5, color='DarkSlateGrey')
                                                ),
                                                legendgroup=f'cluster_{cluster_id}',
                                                showlegend=(idx == 0),
                                                hovertemplate=[t for t, m in zip(regular_texts, cluster_mask) if m]
                                            ),
                                            row=row, col=col
                                        )

                                no_data_mask = [c == -1 for c in cluster_numeric]
                                if any(no_data_mask):
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[x for x, m in zip(regular_x, no_data_mask) if m],
                                            y=[y for y, m in zip(regular_y, no_data_mask) if m],
                                            mode='markers',
                                            name='No data',
                                            marker=dict(
                                                size=marker_size,
                                                color='gray',
                                                line=dict(width=0.5, color='DarkSlateGrey')
                                            ),
                                            legendgroup='no_data',
                                            showlegend=(idx == 0),
                                            hovertemplate=[t for t, m in zip(regular_texts, no_data_mask) if m]
                                        ),
                                        row=row, col=col
                                    )
                            elif color_by_energy_nn and len(valid_colors) > 0:
                                if y_axis_mode == "Energy (eV)":
                                    colorbar_title = "Intensity (# pairs)"
                                    color_min = min([c for c in regular_colors if c is not None])
                                    color_max = max([c for c in regular_colors if c is not None])
                                else:
                                    colorbar_title = "Energy (eV)"
                                    color_min = global_min_energy
                                    color_max = global_max_energy

                                fig.add_trace(
                                    go.Scatter(
                                        x=regular_x,
                                        y=regular_y,
                                        mode='markers',
                                        marker=dict(
                                            size=regular_sizes,
                                            color=regular_colors,
                                            colorscale=nn_colormap.lower(),
                                            showscale=(idx == 0),
                                            cmin=color_min,
                                            cmax=color_max,
                                            colorbar=dict(
                                                title=colorbar_title,
                                                x=1.02,
                                                title_font=dict(size=16),
                                                tickfont=dict(size=14)
                                            ) if (idx == 0) else None,
                                            line=dict(width=0.5, color='DarkSlateGrey')
                                        ),
                                        showlegend=False,
                                        hovertemplate=regular_texts
                                    ),
                                    row=row, col=col
                                )
                            else:
                                fig.add_trace(
                                    go.Scatter(
                                        x=regular_x,
                                        y=regular_y,
                                        mode='markers',
                                        marker=dict(
                                            size=regular_sizes,
                                            color='steelblue',
                                            line=dict(width=0.5, color='DarkSlateGrey')
                                        ),
                                        showlegend=False,
                                        hovertemplate=regular_texts
                                    ),
                                    row=row, col=col
                                )

                        fig.update_xaxes(title_text="Distance (Å)", row=row, col=col, title_font=dict(size=14))
                        y_label = "Energy (eV)" if y_axis_mode == "Energy (eV)" else "Intensity (# pairs)"
                        fig.update_yaxes(title_text=y_label, row=row, col=col, title_font=dict(size=14))

                    neighbor_order_text = f"{selected_neighbor_order}{'st' if selected_neighbor_order == 1 else 'nd'} Nearest Neighbor"
                    fig.update_layout(
                        height=400 * n_rows,
                        title_text=f"{neighbor_order_text} Analysis - Reference: {reference_elem}",
                        title_font_size=20,
                        showlegend=color_by_cluster,
                        plot_bgcolor='white'
                    )

                    st.plotly_chart(fig, )

                    # Info boxes
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.info(f"⭐ Red stars indicate minimum energy structure in each pair")
                    with col_info2:
                        if global_min_energy is not None:
                            st.info(f"Energy range: {global_min_energy:.4f} to {global_max_energy:.4f} eV")
                else:
                    st.warning("Please select at least one element pair to display")


            st.markdown("---")
            st.header("📊 Statistical Analysis: Energy vs. Element Environment")

            if 'energy' in df_nn_filtered.columns and df_nn_filtered['energy'].notna().any():

                unique_structures = df_nn_filtered.groupby('filename').agg({
                    'energy': 'first'
                }).reset_index()

                composition_data = []

                for filename in unique_structures['filename']:
                    structure_pairs = df_nn_filtered[df_nn_filtered['filename'] == filename]
                    energy = structure_pairs['energy'].iloc[0]

                    element_counts = {}
                    element_distances = {}

                    for _, row in structure_pairs.iterrows():
                        pair = row['element_pair']
                        elem1, elem2 = pair.split('-')

                        if elem2 != reference_elem:
                            element_counts[elem2] = element_counts.get(elem2, 0) + row['intensity']
                            if elem2 in element_distances:
                                element_distances[elem2].append(row['distance'])
                            else:
                                element_distances[elem2] = [row['distance']]

                    avg_distances = {elem: np.mean(dists) for elem, dists in element_distances.items()}

                    data_row = {'filename': filename, 'energy': energy}
                    data_row.update({f'{elem}_count': count for elem, count in element_counts.items()})
                    data_row.update({f'{elem}_distance': dist for elem, dist in avg_distances.items()})

                    composition_data.append(data_row)

                df_composition = pd.DataFrame(composition_data).fillna(0)

                count_cols = [col for col in df_composition.columns if col.endswith('_count')]
                distance_cols = [col for col in df_composition.columns if col.endswith('_distance')]

                if len(count_cols) > 0 and len(df_composition) > 1:

                    corr_tabs = st.tabs([
                        "🗺️ Combined Correlation Heatmap",
                        "📊 Element Count Correlations",
                        "📏 Distance Correlations",

                    ])

                    with corr_tabs[0]:
                        st.subheader("Energy Correlation Heatmap")

                        energy_correlations = []
                        feature_names = []

                        for col in count_cols:
                            if df_composition[col].std() > 0:
                                elem = col.replace('_count', '')
                                corr = df_composition['energy'].corr(df_composition[col])
                                energy_correlations.append(corr)
                                feature_names.append(f"{elem}\n(count)")

                        for col in distance_cols:
                            if df_composition[col].std() > 0 and (df_composition[col] > 0).sum() > 1:
                                elem = col.replace('_distance', '')
                                valid_mask = df_composition[col] > 0
                                if valid_mask.sum() > 1:
                                    corr = df_composition.loc[valid_mask, 'energy'].corr(
                                        df_composition.loc[valid_mask, col])
                                    energy_correlations.append(corr)
                                    feature_names.append(f"{elem}\n(distance)")

                        if len(energy_correlations) > 0:
                            corr_array = np.array(energy_correlations).reshape(1, -1)

                            fig_heatmap = go.Figure(data=go.Heatmap(
                                z=corr_array,
                                x=feature_names,
                                y=['Energy'],
                                colorscale='RdBu_r',
                                zmid=0,
                                text=np.round(corr_array, 3),
                                texttemplate='%{text}',
                                textfont={
                                    "size": 18,
                                    "color": "black"
                                },
                                colorbar=dict(
                                    title="Correlation<br>with Energy",
                                    title_font=dict(size=20, color='black'),
                                    tickfont=dict(size=18, color='black'),
                                    thickness=30,
                                    len=0.5
                                ),
                                hovertemplate='%{x}<br>Correlation with Energy: %{z:.3f}<extra></extra>'
                            ))

                            fig_heatmap.update_layout(
                                title={
                                    'text': f"Correlation with Energy: Element Environment around {reference_elem}",
                                    'font': {'size': 24, 'color': 'black'}
                                },
                                xaxis=dict(
                                    tickfont=dict(size=16, color='black'),
                                    tickangle=-45,
                                    side='bottom'
                                ),
                                yaxis=dict(
                                    tickfont=dict(size=18, color='black'),
                                    showticklabels=True
                                ),
                                height=500,
                                width=max(800, len(feature_names) * 80),
                                hoverlabel=dict(
                                    font_size=18,
                                    font_family="Arial",
                                    bgcolor="white",
                                    bordercolor="black"
                                ),
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )

                            st.plotly_chart(fig_heatmap, )


                            st.subheader("Bar Chart View")

                            fig_bar = go.Figure()

                            colors = ['red' if c > 0 else 'blue' for c in energy_correlations]

                            fig_bar.add_trace(go.Bar(
                                x=feature_names,
                                y=energy_correlations,
                                marker_color=colors,
                                text=[f'{c:.3f}' for c in energy_correlations],
                                textposition='outside',
                                textfont=dict(size=16, color='black'),
                                hovertemplate='%{x}<br>Correlation: %{y:.3f}<extra></extra>'
                            ))

                            fig_bar.update_layout(
                                title={
                                    'text': f"Energy Correlations - {reference_elem} Environment",
                                    'font': {'size': 22, 'color': 'black'}
                                },
                                xaxis=dict(
                                    title="Feature",
                                    tickfont=dict(size=16, color='black'),
                                    title_font=dict(size=18, color='black'),
                                    tickangle=-45
                                ),
                                yaxis=dict(
                                    title="Correlation with Energy",
                                    tickfont=dict(size=16, color='black'),
                                    title_font=dict(size=18, color='black'),
                                    zeroline=True,
                                    zerolinewidth=2,
                                    zerolinecolor='black'
                                ),
                                height=700,
                                hoverlabel=dict(
                                    font_size=18,
                                    font_family="Arial"
                                ),
                                plot_bgcolor='white',
                                paper_bgcolor='white'
                            )

                            st.plotly_chart(fig_bar,)


                            with st.expander("📖 How to interpret these correlations"):
                                st.markdown(f"""
                                **Each bar/cell shows correlation between ONE feature and energy:**

                                **Element Count Features:**
                                - **Negative (blue)**: More neighbors of this element → Lower energy (more stable)
                                - **Positive (red)**: More neighbors of this element → Higher energy (less stable)

                                **Distance Features:**
                                - **Negative (blue)**: Larger distance → Lower energy (prefers separated)
                                - **Positive (red)**: Larger distance → Higher energy (prefers close contact)

                                **Strength of correlation:**
                                - **|r| > 0.7**: Very strong relationship
                                - **|r| > 0.5**: Strong relationship
                                - **|r| > 0.3**: Moderate relationship
                                - **|r| < 0.3**: Weak relationship

                                **Example:**
                                - "O (count)" = -0.65: Having more O neighbors strongly stabilizes structures
                                - "Al (distance)" = +0.45: Al being further from {reference_elem} destabilizes (prefers close)
                                """)
                        else:
                            st.warning("No valid correlations to display.")

                    with corr_tabs[1]:
                        st.subheader(f"Element Count vs Energy Correlation")

                        count_correlations = {}
                        for col in count_cols:
                            if df_composition[col].std() > 0:
                                elem = col.replace('_count', '')
                                corr = df_composition['energy'].corr(df_composition[col])
                                count_correlations[elem] = corr

                        if count_correlations:
                            corr_count_df = pd.DataFrame([
                                {
                                    'Element': elem,
                                    'Correlation': f'{corr:.4f}',
                                    'Interpretation': (
                                        'Strong stabilizing' if corr < -0.5 else
                                        'Stabilizing' if corr < -0.2 else
                                        'Destabilizing' if corr > 0.2 else
                                        'Strong destabilizing' if corr > 0.5 else
                                        'Minimal effect'
                                    )
                                }
                                for elem, corr in
                                sorted(count_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                            ])

                            st.dataframe(
                                corr_count_df,
                                hide_index=True,
                                column_config={
                                    "Element": st.column_config.TextColumn(width="medium"),
                                    "Correlation": st.column_config.TextColumn(width="medium"),
                                    "Interpretation": st.column_config.TextColumn(width="large")
                                }
                            )

                            st.caption(f"Higher count = more neighbors of this element around {reference_elem}")
                            st.caption("Positive correlation: more neighbors → higher energy (less stable)")
                            st.caption("Negative correlation: more neighbors → lower energy (more stable)")

                    with corr_tabs[2]:
                        st.subheader(f"Average Distance vs Energy Correlation")

                        distance_correlations = {}
                        for col in distance_cols:
                            if df_composition[col].std() > 0 and (df_composition[col] > 0).sum() > 1:
                                elem = col.replace('_distance', '')
                                valid_mask = df_composition[col] > 0
                                if valid_mask.sum() > 1:
                                    corr = df_composition.loc[valid_mask, 'energy'].corr(
                                        df_composition.loc[valid_mask, col])
                                    distance_correlations[elem] = corr

                        if distance_correlations:
                            corr_dist_df = pd.DataFrame([
                                {
                                    'Element': elem,
                                    'Correlation': f'{corr:.4f}',
                                    'Interpretation': (
                                        'Closer = more stable' if corr > 0.2 else
                                        'Further = more stable' if corr < -0.2 else
                                        'Distance-independent'
                                    )
                                }
                                for elem, corr in
                                sorted(distance_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                            ])

                            st.dataframe(
                                corr_dist_df,

                                hide_index=True,
                                column_config={
                                    "Element": st.column_config.TextColumn(width="medium"),
                                    "Correlation": st.column_config.TextColumn(width="medium"),
                                    "Interpretation": st.column_config.TextColumn(width="large")
                                }
                            )

                            st.caption(f"Average distance between {reference_elem} and neighboring element")
                            st.caption("Positive correlation: larger distance → higher energy (prefers closer)")
                            st.caption("Negative correlation: larger distance → lower energy (prefers further)")
                        else:
                            st.warning("Not enough distance variation to calculate meaningful correlations.")





                    st.markdown("---")
                    st.subheader("📥 Export Correlation Data")

                    col_exp1, col_exp2, col_exp3 = st.columns(3)

                    with col_exp1:
                        csv_comp = df_composition.to_csv(index=False)
                        st.download_button(
                            label="Download Full Data (CSV)",
                            data=csv_comp,
                            file_name=f"composition_distance_around_{reference_elem}.csv",
                            mime="text/csv"
                        )

                    with col_exp2:
                        if 'corr_count_df' in locals():
                            csv_count = corr_count_df.to_csv(index=False)
                            st.download_button(
                                label="Download Count Correlations (CSV)",
                                data=csv_count,
                                file_name=f"count_correlations_{reference_elem}.csv",
                                mime="text/csv"
                            )

                    with col_exp3:
                        if 'corr_dist_df' in locals():
                            csv_dist = corr_dist_df.to_csv(index=False)
                            st.download_button(
                                label="Download Distance Correlations (CSV)",
                                data=csv_dist,
                                file_name=f"distance_correlations_{reference_elem}.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning("Need at least 2 structures with varying compositions for correlation analysis.")


            st.markdown("---")
            st.subheader("Data Preview")
            st.dataframe(df_nn_filtered.head(20), )

            st.subheader("Export Options")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                summary_data = []
                for pair in element_pairs:
                    pair_data = df_nn_filtered[df_nn_filtered['element_pair'] == pair]
                    summary_data.append({
                        'Element_Pair': pair,
                        'Min_Distance_A': pair_data['distance'].min(),
                        'Max_Distance_A': pair_data['distance'].max(),
                        'Total_Data_Points': len(pair_data),
                        'Structures': pair_data['filename'].nunique()
                    })

                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)

                st.download_button(
                    label="Download Summary (CSV)",
                    data=csv_summary,
                    file_name="nn_analysis_summary.csv",
                    mime="text/csv"
                )

            with export_col2:
                if 'corr_df' in locals():
                    csv_corr = corr_df.to_csv(index=False)
                    st.download_button(
                        label="Download Correlations (CSV)",
                        data=csv_corr,
                        file_name="element_energy_correlations.csv",
                        mime="text/csv"
                    )

            with export_col3:
                html_export = fig.to_html() if 'fig' in locals() else None
                if html_export:
                    st.download_button(
                        label="Download Plot (HTML)",
                        data=html_export,
                        file_name="nearest_neighbor_plot.html",
                        mime="text/html"
                    )
with main_tab3:
    st.header("📚 Getting Started Guide")

    st.markdown("""
    Prepare Python script to calculate fingerprints on crystal structures and convert them into 2D space with t-SNE. Perform k-means clustering on the t-SNE or the structures energies. 

    **Access the app here:** [https://fingerprints.streamlit.app/](https://fingerprints.streamlit.app/)  
    **See tutorial at:** [YouTube](https://www.youtube.com/)
    **💻 GitHub Repository:** [https://github.com/bracerino/structures-fingerprints](https://github.com/bracerino/structures-fingerprints)
    """)

    st.markdown("---")

    st.subheader("🔧 Install Prerequisites")

    st.markdown("First, update your system and install required build tools:")

    st.code("""sudo apt update
sudo apt install build-essential python3.12-dev python3-venv""", language="bash")

    st.markdown("---")

    st.subheader("🐍 Set Up Python Virtual Environment")

    st.markdown("Follow these steps to create and activate your virtual environment with all necessary packages:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Step 1: Create Environment**")
        st.code("python3 -m venv finger_env", language="bash")
        st.caption("Creates a new virtual environment named 'finger_env'")

    with col2:
        st.markdown("**Step 2: Activate Environment**")
        st.code("source finger_env/bin/activate", language="bash")
        st.caption("Activates the virtual environment")

    with col3:
        st.markdown("**Step 3: Install Packages**")
        st.code("pip install dscribe numpy matplotlib scikit-learn ase pandas", language="bash")
        st.caption("Installs all required Python packages")

    st.markdown("---")

    st.subheader("▶️ Running the Generated Script")

    st.markdown("""
    After generating your script from the **Generate Script** tab:

    1. Download the generated Python script
    2. Create a folder named `structures` in the same directory
    3. Place your structure files (VASP, CIF, XYZ, etc.) in the `structures` folder
    4. Optionally, create an `energies.txt` file with structure names and energies in the some folder as the Python script
    5. Run the script:
    """)

    st.code("python fingerprint_analysis_YYYYMMDD_HHMMSS.py", language="bash")
