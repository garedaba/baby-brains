import numpy as np
import pandas as pd

import os
from collections import Counter

from scipy.stats import hypergeom

fdr_threshold = 0.05

def main():
    os.makedirs('results/enrichment', exist_ok=True)
    os.makedirs('results/GO', exist_ok=True)

    # LOAD
    # single cell gene data
    all_gene_data = pd.read_csv('data/gene_lists/all-scRNA-data.csv')

    # normalised RPKM bulk data corrected for age, sex, etc
    bulk_data = pd.read_csv('data/processed_psychencode/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2-residualised.csv')

    # gene-wise correlation with PC components
    correlation_results = pd.read_csv('results/gene_correlations/PCA_correlations-KendallTau-residualisedRPKM.csv')

    # fetal background geneset = all filtered genes in bulk data
    background_genes = pd.read_csv('data/gene_lists/background_genes.txt', header=None)[0]
    print('number of background genes: {:}'.format(len(background_genes)))

    # get gene lists
    print('gathering gene lists')
    # genes differentially expressed by classes or categories, returning all genes, as well as those that are unique to each class
    # CELL TIMING: PRECURSOR OR MATURE
    cell_timing, cell_timing_genes, cell_timing_unique_genes = get_gene_lists(all_gene_data, background_genes, class_type='timing')
    # CELL CLASS
    cell_classes, cell_class_genes, cell_class_unique_genes = get_gene_lists(all_gene_data, background_genes, class_type='class')
    # CELL TYPE
    cell_types, cell_type_genes, cell_type_unique_genes = get_gene_lists(all_gene_data, background_genes, class_type='cluster_study')

    # get significant genes
    significant_genes = pd.read_csv('results/gene_correlations/PCA_correlations-KendallTau-PC-significant_genes-p' + str(fdr_threshold) + '.csv')

    # genes positively correlated to PC component
    positive_significant_genes_list = list(significant_genes.loc[significant_genes['PC1_tau']>0,'symbol'])
    # genes negatively correlated to PC component
    negative_significant_genes_list = list(significant_genes.loc[significant_genes['PC1_tau']<0,'symbol'])

    # ENRICHMENT
    print("cell enrichments")
    cell_timing_enrichment_results = run_enrichment(cell_timing, cell_timing_genes, cell_timing_unique_genes, positive_significant_genes_list, negative_significant_genes_list, background_genes)
    cell_timing_enrichment_results.to_csv('results/enrichment/cell_timing_enrichment-PC1-significant_genes-p' + str(fdr_threshold) + '.csv', index=False)
    print("see results/enrichment/cell_timing_enrichment-significant_genes-p" + str(fdr_threshold) + ".csv")

    cell_class_enrichment_results = run_enrichment(cell_classes, cell_class_genes, cell_class_unique_genes, positive_significant_genes_list, negative_significant_genes_list, background_genes)
    cell_class_enrichment_results.to_csv('results/enrichment/cell_class_enrichment-PC1-significant_genes-p' + str(fdr_threshold) + '.csv', index=False)
    print("see results/enrichment/cell_class_enrichment-significant_genes-p" + str(fdr_threshold) + ".csv")

    cell_type_enrichment_results = run_enrichment(cell_types, cell_type_genes, cell_type_unique_genes, positive_significant_genes_list, negative_significant_genes_list, background_genes)
    cell_type_enrichment_results.to_csv('results/enrichment/cell_type_enrichment-PC1-significant_genes-p' + str(fdr_threshold) + '.csv', index=False)
    print("see results/enrichment/cell_type_enrichment-significant_genes-p" + str(fdr_threshold) + ".csv")

    # save out lists for webgestalt
    np.savetxt('results/GO/positive_genes.txt', positive_significant_genes_list, fmt='%s')
    np.savetxt('results/GO/negative_genes.txt', negative_significant_genes_list, fmt='%s')

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

if __name__ == '__main__':
    main()
