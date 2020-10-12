import numpy as np
import pandas as pd

import subprocess, os

from statsmodels.stats.multitest import fdrcorrection
from collections import Counter

# import functions previously defined for cell enrichment
from D__cell_enrichments import get_gene_lists, safe_div, calculate_enrichment, run_enrichment

def main():
    os.makedirs('results/metrics', exist_ok=True)
    # calculate metric differences
    print('')
    print('calculating group differences')
    # load PC coordinates
    pca_data = pd.read_csv('results/PCA/mean-regional-principal-components.csv')
    pca_dict = dict(zip(pca_data['region'], pca_data['PC1']))
    # load corrected cortical metrics for each group
    out_data = pd.read_csv('data/processed_imaging/term-vs-preterm-cortical-metrics-corrected-for-age-and-sex.csv')
    metric_differences = pd.DataFrame(out_data[out_data['term']==1].groupby(['region','metric']).mean()['value'] - out_data[out_data['term']==0].groupby(['region','metric']).mean()['value'])
    metric_differences.reset_index(inplace=True)
    metric_differences['PC1'] = metric_differences['region'].map(pca_dict)
    # save out
    metric_differences.to_csv('results/metrics/metric_differences.csv', index=None)

    # first run windowed correlations: for each gene calculate correlation between estimated expression in a given window
    # and regional group differences in T1/T2 at term
    print("running models in R")
    print("")
    retcode = subprocess.call(['/usr/bin/Rscript','./r_code/windowed_correlations.R'])
    print("")
    if retcode==1:
        ("something went wrong...")

    # load windowed correlations between gene expression and myelin differences
    windowed_correlations = pd.read_csv('results/gene_correlations/windowed_correlations.csv')

    # get cell class lists
    background_genes = pd.read_csv('data/gene_lists/background_genes.txt', header=None)[0]
    all_gene_data = pd.read_csv('data/gene_lists/all-scRNA-data.csv')
    cell_classes, cell_class_genes, cell_class_unique_genes = get_gene_lists(all_gene_data, background_genes, class_type='class')

    # calculate cell class enrichment in each age window
    print('calculating enrichment of cell class genes in significantly correlated genes for each age window')
    siggenes=[]
    enrich=[]
    p=[]
    for w in np.arange(10):
        # for each window
        win=w+1
        selected_window = windowed_correlations[windowed_correlations['age']==win]
        pval = selected_window['pval'].values

        # adjust p-value for FDR across genes
        adj_pval = fdrcorrection(pval)[1]

        # get significant genes (p<0.05)
        significant_genes = selected_window.loc[adj_pval<.05,['symbol', 'tau']]

        # genes postively correlated to group difference - higher expression in regions that are most different
        positive_significant_genes_list = significant_genes.loc[significant_genes['tau']>0,:]
        negative_significant_genes_list = significant_genes.loc[significant_genes['tau']<0,:]

        # keep list of sig genes in each window
        siggenes.append(positive_significant_genes_list['symbol'].values)

        # run class enrichment
        cell_class_enrichment_results = run_enrichment(cell_classes, cell_class_genes, cell_class_unique_genes, list(positive_significant_genes_list['symbol']), list(negative_significant_genes_list['symbol']), background_genes)
        # keep only results for genes positively correlated (all genes in a cell class) with group differences (and therefore T1/T2)
        cell_class_enrichment_results = cell_class_enrichment_results[(cell_class_enrichment_results['gene_list']=='all') & (cell_class_enrichment_results['loading']=='positive')]

        enrich.append((cell_class_enrichment_results['enrichment']))
        p.append((cell_class_enrichment_results['p']))

    # stack up results into dataframe
    windowed_enrichment = pd.DataFrame(np.vstack(enrich))
    windowed_enrichment.columns = cell_classes
    windowed_p = pd.DataFrame(np.vstack(p))
    windowed_p.columns = [i + '_p' for i in cell_classes]

    windowed_enrichment = pd.DataFrame(np.vstack(enrich))
    windowed_enrichment.columns = cell_classes
    windowed_p = pd.DataFrame(np.vstack(p))
    windowed_p.columns = [i + '_p' for i in cell_classes]

    # concatenate enrichment and p-values
    out_csv = pd.concat((windowed_enrichment, windowed_p), axis=1)
    out_csv = out_csv[sorted(out_csv.columns)]
    out_csv = out_csv.reset_index()
    out_csv.rename(columns={'index':'age window'}, inplace=True)

    # save out
    print('see: results/enrichment/windowed-enrichment.csv')
    print('')
    out_csv.to_csv('results/enrichment/windowed-enrichment.csv', index=False)

if __name__ == '__main__':
    main()
