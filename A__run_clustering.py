#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

from scipy.cluster import hierarchy
from scipy.stats import zscore
from scipy.spatial.distance import pdist

from sklearn.metrics import silhouette_score

#### SETUP #####################################################################
n_samples = 10000
n_perms = 10000
################################################################################

#### HELPERS ###################################################################
def bootstrap_silhouette_score(data, n_clusters, samples=100):
    bootstrap_score = np.zeros((len(n_clusters), samples))
    
    g = data.groupby(['ids'])

    bootstrapped_data = []
    for s in np.arange(samples):
        # get random sample with replacement
        sample = np.random.choice(g.ngroups, size=(g.ngroups), replace=True)

        # get index of all records for selected subjects
        sampInd = np.hstack(np.vstack(list(g.groups.values()))[sample,:])

        # select data
        bootstrap_data = data.loc[sampInd]
        
        # group average
        mean_bootstrap_data = bootstrap_data.groupby('metric').mean().loc[:,'A1C':'VLPFC']
        # save bootstrapped mean for later
        bootstrapped_data.append(mean_bootstrap_data)
        
        # normalise and cluster
        bootM = mean_bootstrap_data.T.apply(zscore, ddof=1, axis=0)
        bootstrap_row_linkage = hierarchy.linkage(bootM, 
                                    metric='cosine', method='average')
    
        for n, c in enumerate(n_clust):
            bootstrap_score[n, s] = silhouette_score(bootM, hierarchy.fcluster(bootstrap_row_linkage, c, criterion='maxclust'), metric='cosine')

    return bootstrap_score, bootstrapped_data
##################################################################################

### RUN ##########################################################################
os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

print("")
print("loading mean regional data")
# load data
scaled_clean_data = pd.read_csv('data/regional-multimodal-metrics-term-data-clean-zscore.csv')
mean_regional_data = pd.read_csv('data/regional-multimodal-metrics-term-data-clean-zscore-groupaverage.csv', index_col=0)
M = mean_regional_data.T.apply(zscore, ddof=1, axis=0)

print("clustering...")

row_linkage = hierarchy.linkage(
    M, metric='cosine', method='average')

col_linkage = hierarchy.linkage(
    M.T, metric='cosine', method='average')

c_corr =  hierarchy.cophenet(row_linkage, pdist(M, metric='cosine'))[0]

print('cophenetic correlation: %0.2f' % c_corr)
print("")

# cluster IDs
four_cluster_ids = hierarchy.fcluster(row_linkage, 4, criterion='maxclust')
cluster_data = pd.DataFrame([mean_regional_data.T.index,four_cluster_ids]).T
cluster_data.columns = ['region', 'FourCluster']
cluster_data = cluster_data.sort_values('FourCluster')
cluster_data.to_csv('results/regional-clusters.csv', index=True)
print("Four cluster solution: see results/regional-clusters.csv")
print("")

# bootstrap silhouette score estimates
print("bootstrapping...")
n_clust = [2,3,4,5,6,7]

bootstrap_scores, bootstrapped_data = bootstrap_silhouette_score(scaled_clean_data, n_clust, samples=n_samples)
bootstrap_scores = pd.DataFrame(bootstrap_scores.T)
bootstrap_scores.columns = n_clust

bootstrap_summary = pd.DataFrame()
bootstrap_summary['clusters'] = n_clust
bootstrap_summary['mean'] = bootstrap_scores.mean(axis=0).values
bootstrap_summary['std'] = bootstrap_scores.std(axis=0).values
bootstrap_summary['upper'] = bootstrap_scores.quantile(q=0.975).values
bootstrap_summary['lower'] = bootstrap_scores.quantile(q=0.025).values


## PLOTS ##############################################
print("")
print("plotting")
# plot
sns.set_context('poster')
g = sns.clustermap(M, row_linkage=row_linkage,col_linkage=col_linkage, cmap="Greens_r", figsize=(8,10))
g.cax.set_visible(False)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
for a in g.ax_row_dendrogram.collections:
    a.set_linewidth(2)

for a in g.ax_col_dendrogram.collections:
    a.set_linewidth(2)
g.savefig('graphs/Figure1B.pdf')
print("see graphs/Figure1B.pdf")

# plot
sns.set_context('paper')
plt.figure(figsize=(10,6))
plt.plot(n_clust, bootstrap_summary['mean'], color='#218708')
plt.fill_between(n_clust, bootstrap_summary['upper'], y2=bootstrap_summary['lower'], alpha=0.2, color='#218708')
plt.text(1.8,.315,'10000 bootstraps', fontsize=16)

plt.ylabel('Silhouette score', fontsize=20)
plt.xlabel('Cluster number', fontsize=20)
plt.xticks(ticks=n_clust)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('graphs/FigureS2.pdf')
print("see graphs/FigureS2.pdf")

# plot
sns.set_context('talk')
# use subject metrics
long_data = pd.melt(scaled_clean_data, id_vars=['ids','sess','age_at_scan','age_at_birth','male','metric'],
                    var_name='roi', value_name='Z-scored metric')
long_data['cluster']=long_data['roi'].map(dict(zip(cluster_data['region'],cluster_data['FourCluster'])))

g = sns.catplot(x="roi", y="Z-scored metric", 
            hue="cluster", col="metric", 
            order = mean_regional_data.columns[cluster_data.index.values],
            data=long_data,
            col_wrap=3, 
            sharex=True, sharey=True,
            palette='Paired', jitter=1, alpha=.8, s=6, edgecolor='w', linewidth=1)
sns.despine(top=False, right=False)
g.set_xticklabels(labels=cluster_data['region'],rotation=45)
g.set_titles(row_template = '{row_name}', col_template = '{col_name}')
g.savefig('graphs/Figure1C.pdf')
print("see graphs/Figure1C.pdf")
