#!/usr/bin/env python
# coding: utf-8

# In[58]:


import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDistGeom
IPythonConsole.ipython_3d = True


from openbabel import openbabel
from openbabel import pybel
import sys
import os
import openbabel as ob
import numpy as np
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from ase.io import read
from dscribe.descriptors import SOAP


# In[3]:


from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import vq
import shutil
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


# In[4]:


# Load the SDF file
sdf_file = "clevin.sdf"  # Replace with the path to your SDF file

# Read the entire SDF file as a list of lines
with open(sdf_file, 'r') as file:
    lines = file.readlines()

# Initialize lists to store conformers and energies
conformers = []
energies = []

current_conformer = []
current_energy = None

# Process the file line by line
for line in lines:
    if line.startswith("$$$$"):  # End of a conformer
        if current_conformer and current_energy is not None:
            conformers.append(current_conformer)
            energies.append(current_energy)
        current_conformer = []
        current_energy = None
    else:
        current_conformer.append(line)
        if "Energy:" in line:
            current_energy = float(line.split()[3])

# Find the lowest energy conformer
min_energy = min(energies)


# Filter conformers within 10 kcal/mol from the lowest energy conformer
energy_threshold = 10  # kcal/mol
filtered_conformers = []
filtered_energies = []

for i, energy in enumerate(energies):
    if energy - min_energy <= energy_threshold:
        filtered_conformers.append((i, conformers[i]))
        filtered_energies.append(energy)

# Output the number of filtered conformers
num_filtered_conformers = len(filtered_conformers)
print(f"Number of conformers within {energy_threshold} kcal/mol: {num_filtered_conformers}")


# In[9]:


# Now we generate separate SDF files and record the energies
sdf_files_and_energies = []
for idx, (i, conformer) in enumerate(filtered_conformers):
    output_filename = f"clevin/conformer_{i}.sdf"
    with open(output_filename, 'w') as outfile:
        for line in conformer:
            outfile.write(line)
        outfile.write("$$$$\n")
    sdf_files_and_energies.append((output_filename, filtered_energies[idx]))


# In[65]:


############# SOAP features ###############
soap_features = []
soap = SOAP(
        species=['Na', 'H', 'C', 'N'],
        periodic=False,
        r_cut=10.0, ## distance of sphere sampled from centre in angstrom
        sigma=0.1,
        n_max=3,
        l_max=7,
    )

for i in range(num_conformers):
    print(i)
    atoms = read(os.path.join(temp_dir, f"conformer_{i+1}.xyz"))
    #print(smile)
    #unique_elements = list(set(atoms.get_chemical_symbols()))
    #print(unique_elements)
    #si_indices = [atom.index for atom in atoms if atom.symbol == 'Si']
    #print(si_indices)
    temp_soap = soap.create(atoms, centers=[129])  ### ase starts index with 0 and gaussian16 starts with 1
    soap_features.append(temp_soap[0])
print(len(soap_features[0]))
print(range(num_conformers))


# import subprocess
# import os
# import numpy as np
# 
# def calculate_rmsd_matrix(filtered_conformer_list, temp_dir):
#     """
#     Calculate the RMSD matrix between conformers in SDF format.
#     1. Converts SDF to XYZ using Open Babel.
#     2. Uses `calculate_rmsd` command-line tool to calculate RMSD.
#     """
#     num_conformers = len(filtered_conformer_list)
#     rmsd_matrix = np.zeros((num_conformers, num_conformers))  # Initialize RMSD matrix
#     
#     # Write conformer i to an SDF file
#     #for i in range(num_conformers):
#     #    sdf1_filename = os.path.join(temp_dir, f"conformer_{i+1}.sdf")
#     #    xyz1_filename = os.path.join(temp_dir, f"conformer_{i+1}.xyz")
# 
#         # Write conformer i to an SDF file
#     #    with open(sdf1_filename, 'w') as f:
#     #        f.writelines(filtered_conformer_list[i])
# 
#         # Convert SDF to XYZ using Open Babel
#     #    subprocess.run(['obabel', sdf1_filename, '-O', xyz1_filename], check=True)
# 
#     # compute rmsd for conf i and j
#     for i in range(num_conformers):
#         sdf1_filename = os.path.join(temp_dir, f"conformer_{i+1}.sdf")
#         xyz1_filename = os.path.join(temp_dir, f"conformer_{i+1}.xyz")
#         print(str(i)+"...")
#         for j in range(i + 1, num_conformers):
#             sdf2_filename = os.path.join(temp_dir, f"conformer_{i+1}.sdf")
#             xyz2_filename = os.path.join(temp_dir, f"conformer_{i+1}.xyz")
#             # Calculate RMSD using the command-line tool
#             proc = subprocess.check_output(['calculate_rmsd', '--no-hydrogen', '--reorder', xyz1_filename, xyz2_filename],stderr=subprocess.STDOUT)
#             print(str(j)+",")
#             rmsd_value = float(proc)
#             # Store RMSD in the matrix (symmetric)
#             rmsd_matrix[i, j] = rmsd_value
#             rmsd_matrix[j, i] = rmsd_value  # Symmetric matrix
# 
#     return rmsd_matrix
# 

# # Create a temporary directory for saving intermediate SDF and XYZ files
# temp_dir = "temp_conformers/"
# os.makedirs(temp_dir, exist_ok=True)
# 
# # Calculate the RMSD matrix
# rmsd_matrix = calculate_rmsd_matrix(filtered_conformer_list, temp_dir)
# 
# # Output the RMSD matrix
# print("RMSD Matrix:")
# print(rmsd_matrix)
# 

# np.savetxt('clevin_rmsd_matrix.txt', rmsd_matrix, fmt='%d')

# In[ ]:


sns.reset_defaults()


# In[ ]:


vmin = 0.0
vmax = 1.0
annot = False

cmap = plt.get_cmap("Blues")
plot = sns.clustermap(data = rmsd_matrix,
                      vmin =vmin, 
                      vmax = vmax, 
                      cmap = cmap, 
                      linewidth=.5, 
                      annot=annot,
                      
                      figsize=(5, 5),
                     dendrogram_ratio=(.1, .1),
                     cbar_pos=(1.1, 0.08, .03, 0.8),
                      #yticklabels=False
                     )
plot.ax_row_dendrogram.set_visible(False)
ids = plot.dendrogram_row.reordered_ind
print(ids)

plt.savefig('plot.pdf', dpi=600)
plt.show()


# In[71]:


from scipy.cluster.vq import vq
from umap import UMAP  # Added UMAP import

features = soap_features

# 1. Standardize the features for KMeans
scaled_features = StandardScaler().fit_transform(features)

# 2. Define KMeans arguments for clustering and SSE calculations
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
}

# 3. Calculate SSE for a range of cluster values to find the elbow
sse = [KMeans(n_clusters=k, **kmeans_kwargs).fit(scaled_features).inertia_ for k in range(1, 100)]
kl = KneeLocator(range(1, 100), sse, curve="convex", direction="decreasing")
print ('best cluster number =' + str(kl))

# 4. Perform KMeans clustering with the optimal number of clusters from KneeLocator
kmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs).fit(scaled_features)

# 5. Extract cluster labels and print the number of iterations and cluster centers
print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")



# In[72]:


print (kl.elbow)


# In[76]:


# 6. Use UMAP to reduce dimensionality for visualization
umap_embedding = UMAP(n_components=2, random_state=42).fit_transform(scaled_features)

# 7. Assign closest points to cluster centers and calculate distances
closest, distances = vq(kmeans.cluster_centers_, scaled_features)
print(f"Closest points: {closest}\nDistances: {distances}")

num_labels = len(set(kmeans.labels_))  # Count unique labels
palette = sns.color_palette("tab20", num_labels)  # Use a large enough palette (tab20 can handle 20 distinct labels)


# 8. Extract UMAP coordinates for plotting
data = pd.DataFrame({
    "x": umap_embedding[:, 0], 
    "y": umap_embedding[:, 1], 
    "label": kmeans.labels_
})
data.to_csv("/Users/mantingmu/Desktop/test.csv")

# 9. Plot the clustered points using seaborn and UMAP
sns.scatterplot(data=data, x="x", y="y", hue="label", palette=palette)
plt.savefig('/Users/mantingmu/Desktop/umap.pdf', dpi=300)
plt.show()


# In[78]:


# 10. closest conformer files
for c in closest:
    print (c+1)


# In[83]:


data = pd.DataFrame({
    "index": range(1, len(kmeans.labels_) + 1),  # Create 1-based index
    "x": umap_embedding[:, 0],
    "y": umap_embedding[:, 1],
    "label": kmeans.labels_
})

sorted_data = data.sort_values(by='label')

# Print the index and corresponding K-means labels
for idx, x, y, label in zip(sorted_data['index'], sorted_data['x'], sorted_data['y'], sorted_data['label']):
    print(f"Index: {idx}, K-means label: {label}")


# In[ ]:




