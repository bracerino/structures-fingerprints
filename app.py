import streamlit as st
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
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

main_tab1, main_tab2, main_tab3 = st.tabs(["Generate Script", "Interactive Visualization: 2D t-SNE Map", "Getting Started Guide"])

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
            save_fingerprints_code = f"""
print("Saving fingerprints...")
np.save("{fingerprints_filename}", np.array(fingerprints))
print(f"  Fingerprints saved to {fingerprints_filename}")
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

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Export Options")

        html_str = fig.to_html()
        st.download_button(
            label="Download Interactive Plot (HTML)",
            data=html_str,
            file_name="tsne_interactive.html",
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
