#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 

from scipy.stats import zscore


### HELPERS ######################################################################################################
def pc_plot(data, palette, hue=None, ax=None):
   
    
    if ax==None:
        ax = plt.gca()
    g = sns.barplot(np.arange(11), data, dodge=False,
            hue=hue, palette=palette, ax=ax)
    sns.despine(bottom=True, left=True)

    ax.axhline(0, color='black')
    g.tick_params(axis='both', labelsize=20, length = 0)
    
    l = g.legend()
    l.remove()
    
    
def pc_scatter(x, y, palette, hue, ax=None):      
    if ax==None:
        ax = plt.gca()
    g = sns.scatterplot(x=x, y=y, 
                hue=hue, palette=palette, s=100)

    g.set_xlim(-4,4)
    g.set_xlabel('PC1', fontsize=20)
    g.set_ylim(-3.5,3.5)
    g.set_ylabel('PC2', fontsize=20)
    l = g.legend()
    l.remove()
    plt.tick_params(axis='both', which='major', labelsize=20, length=0)

    plt.text(1.28821666, 0.1786681, 'DLPFC', fontsize=12)
    plt.text(2.12842571, -0.38226916, 'IPC', fontsize=12)
    plt.text(2.89700453, 0.2603563, 'ITC', fontsize=12)
    plt.text(1.92268099, 1.04029276, 'OFC', fontsize=12)
    plt.text(0.64756212, -0.18158481, 'VLPFC', fontsize=12)
    plt.text(0.20445649, 0.52690845, 'MFC', fontsize=12)
    plt.text(0.18186962, 1.1862678 , 'STC', fontsize=12)
    plt.text(-3.53813618, -0.33143701, 'A1C', fontsize=12)
    plt.text(-2.28478104, -0.4155017 , 'M1', fontsize=12)
    plt.text(-3.28016872,  1.08533431, 'S1', fontsize=12)
    plt.text(0.4428698 , -2.95820467, 'V1', fontsize=12)   
        

def run_PCA(data):
    cov_matrix = np.cov(data, rowvar=False, ddof=1)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    ix = np.argsort(-eig_vals)

    eig_vals = eig_vals[ix]
    eig_vecs= eig_vecs[:,ix]

    eigen_vectors = eig_vecs[:,:]

    # principal components
    PCs = np.dot(data, eigen_vectors)
    
    return PCs, eigen_vectors, eig_vals


#################################################################################################################

### RUN #########################################################################################################

# LOAD
cluster_data = pd.read_csv('results/regional-clusters.csv', index_col=0)
mean_regional_data = pd.read_csv('data/regional-multimodal-metrics-term-data-clean-zscore-groupaverage.csv', index_col=0)
    
# PCA with mean regional data
print("")
print("performing PCA on mean regional data")
M = mean_regional_data.T  # rotate to n_regions on metric axes (region x metric)
M = zscore(M, axis=0) 

# eigendecomposition of cov matrix
mean_PCs, mean_eigen_vectors, eig_vals = run_PCA(M)

metric_vectors = pd.DataFrame([mean_regional_data.index.values, mean_eigen_vectors[:,0], mean_eigen_vectors[:,1]]).T
metric_vectors.columns = ['metric', 'PC1', 'PC2']
metric_vectors.to_csv('results/mean-regional-principal-components-eigenvectors.csv')
print('see: results/mean-regional-principal-components-eigenvectors.csv')

pc_data = cluster_data.loc[:,['region', 'FourCluster']]
pc_data['PC1'] = mean_PCs[cluster_data.index,0]
pc_data['PC2'] = mean_PCs[cluster_data.index,1]
pc_data.to_csv('results/mean-regional-principal-components.csv')
print("see: results/mean-regional-principal-components.csv")

print("")
print("plotting")

################################################################################################################

### PLOTS ######################################################################################################
palette=['#abcbda', '#3673a0','#b1d394','#42913c']
# PLOT 1
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12,10))

pc_plot(mean_PCs[cluster_data.index,0], palette, hue=cluster_data['FourCluster'], ax=ax1)
pc_plot(mean_PCs[cluster_data.index,1], palette, hue=cluster_data['FourCluster'], ax=ax2)

plt.xticks(ticks=np.arange(11), labels=cluster_data['region'], fontsize=20)
ax1.text(-0.5, -3.4, 'explained variance: ' + '%0.1f' % (eig_vals[0]/sum(eig_vals)*100) + '%', fontsize=16)
ax2.text(-0.5, -3.4, 'explained variance: ' + '%0.1f' % (eig_vals[1]/sum(eig_vals)*100) + '%', fontsize=16)
ax1.set_ylabel('PC1', fontsize=20)
ax2.set_ylabel('PC2', fontsize=20)

plt.savefig('graphs/Figure1D.pdf')
print("see:graphs/Figure1D.pdf")

# PLOT 2
fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,5))

x = mean_PCs[cluster_data.index,0]
y = mean_PCs[cluster_data.index,1]
hue = cluster_data['FourCluster']

pc_scatter(x,y, palette, hue, ax=ax1)
plt.tight_layout()
plt.savefig('graphs/Figure1E.pdf')
print("see:graphs/Figure1E.pdf")




