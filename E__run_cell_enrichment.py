#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from statsmodels.stats.multitest import fdrcorrection
from collections import Counter
from scipy.stats import hypergeom

### HELPERS ##############################################################################################
def get_gene_lists(data, background_genes, class_type='class'):
    # ALL GENES EXPRESSED IN A GIVEN CELL CLASS
    class_genes = []
    classes = np.unique(data[class_type])

    # remove neuron class if it's there (not split into excit. and inhib.),
    classes = list(set(classes) - set(['neuron']))

    for cell_class in classes:
        # genes in cell class
        class_data = np.array(data.loc[data[class_type]==cell_class, 'gene'].values, dtype=str).reshape(-1)
        # only keep genes that are also present in bulk data
        class_data = class_data[pd.DataFrame(class_data).isin(list(background_genes)).values.reshape(-1)]
        class_genes.append(np.unique(class_data))
    
    # ALL GENES *UNIQUELY* EXPRESSED IN A GIVEN CELL CLASS
    ctr = Counter(np.hstack(class_genes))
    gene_count = pd.DataFrame([list(ctr.keys()), list(ctr.values())]).T
    repeat_genes = gene_count.loc[(gene_count[1]>1),0].values

    class_unique_genes = class_genes.copy()
    for n,cell_class in enumerate(classes):
        # remove shared
        class_unique_genes[n] = list(set(class_unique_genes[n])-set(repeat_genes))
        
    return classes, class_genes, class_unique_genes


def safe_div(x,y):
    if y == 0:
        return np.array([0])
    return x / y


def calculate_enrichment(hit_list, top_genes, full_gene_list):
    x = sum(pd.DataFrame(top_genes).isin(hit_list).values) # how many top genes in cell list
    n = sum(pd.DataFrame(hit_list).isin(full_gene_list).values)[0] # how many cell genes in full list
    N = len(top_genes)  # number of samples
    M = len(full_gene_list)  # total number in population
    
    enrichment = safe_div( (x/N) , ((n-x) / (M-N)) ) 
    p = hypergeom.sf(x-1, M, n, N)

    return enrichment, p


def run_enrichment(classes, gene_lists, unique_gene_lists, positive_genes, negative_genes, background_genes):

    enrichment_results = []
    num_genes = []
    # for each cell class/type
    for i in np.arange(len(classes)):
	# for full and unique gene lists
        for gl in [gene_lists, unique_gene_lists]:
            # as long as there are some genes
            if len(gl[i])>0:
                # calculate enrichment in the postively and negatively correlated lists
                for g in [positive_genes, negative_genes]:
                    enrichment_results.append(calculate_enrichment(list(gl[i]), list(g), list(background_genes)))
                    num_genes.append(len(gl[i]))
	    # otherwise is nan	
            else: 
                for g in [positive_genes, negative_genes]:
                    enrichment_results.append((np.array([np.nan]),np.array([np.nan])))
                    num_genes.append(np.array([0]))

    # collate into dataframe
    results = pd.DataFrame(np.hstack(enrichment_results).T)
    results.columns=['enrichment', 'p']
    results['class'] = np.repeat(classes, 4)
    results['loading'] = ['positive', 'negative']*(len(classes)*2)
    results['gene_list'] = np.hstack([np.repeat(['all', 'unique'], 2)]*len(classes))
    results['num_genes'] = np.squeeze(num_genes)
    results = results.loc[:,['class','loading','gene_list','num_genes','enrichment','p']]
    
    return results

##########################################################################################################

fdr_threshold = 0.05

# LOAD
# single cell gene data
all_gene_data = pd.read_csv('data/all-scRNA-data.csv')

# normalised RPKM bulk data corrected for age, sex, etc
bulk_data = pd.read_csv('data/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2-residualised.csv')

# gene-wise correlation with PC components
significant_genes = pd.read_csv('results/PCA_correlations-KendallTau-PC-significant_genes-p' + str(fdr_threshold) + '.csv')

# background geneset = all filtered genes in bulk data
background_genes = np.unique(bulk_data['symbol'])

# get gene lists
print('gathering gene lists')
cell_classes, cell_class_genes, cell_class_unique_genes = get_gene_lists(all_gene_data, background_genes, class_type='class')

# save out cell_class gene lists
for n,c in enumerate(cell_classes):
    cell_genes = pd.DataFrame(cell_class_genes[n])
    cell_genes.columns=['genes']
    cell_genes.to_csv('data/' + c + '-gene-list.csv', index=False)
    cell_unique_genes = pd.DataFrame(cell_class_unique_genes[n])
    cell_unique_genes.columns=['genes']
    cell_unique_genes.to_csv('data/' + c + '-unique-gene-list.csv', index=False)


print("")
print('running enrichments')

# genes postively correlated to PC component
positive_significant_genes_list = list(significant_genes.loc[significant_genes['PC1_tau']>0,'symbol'])
# genes postively correlated to PC component
negative_significant_genes_list = list(significant_genes.loc[significant_genes['PC1_tau']<0,'symbol'])

# ENRICHMENT
print("cell enrichments")
cell_class_enrichment_results = run_enrichment(cell_classes, cell_class_genes, cell_class_unique_genes, positive_significant_genes_list, negative_significant_genes_list, background_genes)
cell_class_enrichment_results.to_csv('results/cell_class_enrichment-PC-significant_genes-p' + str(fdr_threshold) + '.csv', index=False)
print("see results/cell_class_enrichment-PC-significant_genes-p" + str(fdr_threshold) + ".csv")

### PLOTTING ##########################################################################################################################################
print("")
print("plotting")

sns.set_style(style='white')
cell_colours = sns.color_palette([(0.65, 0.46, 0.11), 
                                  (0.10, 0.62, 0.46),
                                  (0.91, 0.16, 0.54),
                                  (0.21, 0.22, 0.23),
                                  (0.45, 0.44, 0.70),
                                  (0.85, 0.37, 0.01),
                                  (0.31, 0.32, 0.33),
                                  (0.40, 0.65, 0.12),
                                  (0.90, 0.67, 0.01),
                                  (0.40, 0.30, 0.21)])
cell_name_dict = {'neuron_excitatory':'excitatory neuron',
                  'neuron_inhibitory':'inhibitory neuron',
                  'radial_glia':'radial glia',
                  'neuron':'neuron: unspecified'}

positive_enrichment = cell_class_enrichment_results.loc[cell_class_enrichment_results.loading=='positive']
positive_enrichment = positive_enrichment[positive_enrichment.gene_list=='all'].copy()
positive_enrichment['class'] = positive_enrichment['class'].map(cell_name_dict).fillna(positive_enrichment['class'])

negative_enrichment = cell_class_enrichment_results.loc[cell_class_enrichment_results.loading=='negative']
negative_enrichment = negative_enrichment[negative_enrichment.gene_list=='all'].copy()
negative_enrichment['class'] = negative_enrichment['class'].map(cell_name_dict).fillna(negative_enrichment['class'])

order = ['astrocyte', 'progenitor', 'excitatory neuron', 'endothelial', 'inhibitory neuron', 'radial glia', 'microglia', 'OPC', 'oligodendrocyte', 'pericyte']

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,7), sharey=False, sharex=True)
sns.barplot(y='class', x='enrichment', hue='class', hue_order=order, palette=cell_colours, data=positive_enrichment,
            dodge=False, ax=ax1, orient='h')

sns.barplot(y='class', x='enrichment', hue='class', hue_order=order, palette=cell_colours, data=negative_enrichment,
            dodge=False, ax=ax2, orient='h')

for ax in [ax1,ax2]:
    ax.legend().set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('')
    ax.set_xlabel('enrichment ratio', fontsize=20)
ax1.set_title('positive genes', fontsize=20)
ax2.set_title('negative genes', fontsize=20)

plt.tight_layout()
plt.savefig('graphs/Figure3C.pdf')
print("see graphs/Figure3C.pdf")
