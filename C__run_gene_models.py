#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import subprocess

from statsmodels.stats.multitest import fdrcorrection

### RUN R MODELS ###################################################################################################
print("running models in R")
print("")
retcode = subprocess.call(['/usr/bin/Rscript','./r_code/nonlinear_models.R'])
print("")
if retcode==1:
	print("something went wrong...")

print('For gene/PC1 correlations: see results/PCA_correlations-KendallTau-residualisedRPKM.csv')
print('For cortical average gene trajectories over gestation: see results/gene-trajectories.csv')
######################################################################################################################


### GET SIGNIFICANT GENE LISTS ######################################################################################
correlation_results = pd.read_csv('results/PCA_correlations-KendallTau-residualisedRPKM.csv')

fdr_threshold=0.05
# get p-values
pval = correlation_results['PC1_pval'].values  
# adjust for FDR
adj_pval = fdrcorrection(pval)[1]
# get significant genes
significant_genes = correlation_results.loc[adj_pval<fdr_threshold,['symbol', 'PC1_tau']]
    
# genes postively correlated to PC component
positive_significant_genes_list = list(significant_genes.loc[significant_genes['PC1_tau']>0,'symbol'])
# genes postively correlated to PC component
negative_significant_genes_list = list(significant_genes.loc[significant_genes['PC1_tau']<0,'symbol'])

# save out
# ranked gene lists
correlation_results.loc[:,['symbol', 'PC1_tau']].sort_values(by='PC1_tau', ascending=False).to_csv('results/PCA_correlations-KendallTau-PC-ranked_list.csv', index=False)
# significant genes only
pd.DataFrame(significant_genes).to_csv('results/PCA_correlations-KendallTau-PC-significant_genes-p' + str(fdr_threshold) + '.csv', index=False)
print(len(significant_genes), 'genes with expression correlated to PC : FDR-corrected at ' + str(fdr_threshold))
print("")
print("see: results/PCA_correlations-KendallTau-PC-ranked_list.csv")
print("see: results/PCA_correlations-KendallTau-PC-significant_genes-p" + str(fdr_threshold) + ".csv")
print("")
