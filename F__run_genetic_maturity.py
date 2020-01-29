#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import zscore, spearmanr, pearsonr

import matplotlib.pyplot as plt
import seaborn as sns



### HELPERS ####################################################################################################

def pivot_bulk_data(data, values='log2_rpkm'):
    # select data
    selected_data = data.loc[:,['symbol', 'sample', 'region', 'age', values]]
    # pivot 
    table_data = selected_data.pivot_table(index=['sample', 'region', 'age'], columns='symbol', values=values)
    table_data = table_data.reset_index()
    
    return table_data

def predict_age_difference(train_data, test_data, model, n_boot=10, permute_idx=None):

    # extract gene cortical expression data (averaged across all ROI)
    mean_training_data = training_data.groupby(['sample', 'age']).mean().reset_index()
    # extract gene columns
    mean_training_data_genes = mean_training_data.drop(['sample','age','PC'], axis='columns')
    # extract age columns
    mean_training_data_age = mean_training_data['age']

    # gene expression for test data
    testing_genes = testing_data.drop(['sample','age','region','PC'], axis='columns')
    n_genes = np.shape(testing_genes)[1]

    # bootstrap genes for model
    pred = np.zeros((len(testing_genes), n_boot))
    corr = np.zeros((n_boot))
    for b in np.arange(n_boot):
        idx = np.random.choice(np.arange(n_genes), size=n_genes, replace=True)

        # fit model with bootstrapped genes
        if permute_idx is None:
            model.fit(mean_training_data_genes.iloc[:,idx], mean_training_data_age)
            pred[:,b] = model.predict(testing_genes.iloc[:,idx]) + np.mean(mean_training_data_age)
            corr[b] = pearsonr(testing_data['PC'], pred[:,b])[0]
        else:
            model.fit(mean_training_data_genes.iloc[permute_idx,idx], mean_training_data_age)
            pred[:,b] = model.predict(testing_genes.iloc[:,idx]) + np.mean(mean_training_data_age)
            corr[b] = pearsonr(testing_data['PC'], pred[:,b])[0]

    return pred, corr

################################################################################################################

### LOAD DATA ##################################################################################################
ss = StandardScaler()
kr = KernelRidge(kernel='linear') # default C
pipe = Pipeline([('scaler',ss), ('regressor', kr)])

# load sample, sex, RIN corrected data
bulk_data = pd.read_csv('results/gene-data-corrected.csv')
significant_genes = pd.read_csv('results/PCA_correlations-KendallTau-PC-significant_genes-p0.05.csv')['symbol']

# load PC data for later
pca_data = pd.read_csv('results/mean-regional-principal-components.csv')
pca_dict = dict(zip(pca_data['region'], pca_data['PC1']))

# pivot table
model_data = pivot_bulk_data(bulk_data, values='residuals')  # sample data, all genes
model_data.insert(3, 'PC', model_data['region'].map(pca_dict))

# select significant genes
significant_model_data = pd.concat((model_data[['sample','region','age', 'PC']], model_data[significant_genes]), axis=1)

# number of bootstraps
n_boot = 5000

################################################################################################################

### MODELLING ##################################################################################################
# use mean cortical data to predict regional age based on gene expression
# calculate correlation between PC1 and predicted age errors

# use both sig. genes and all genes datasets
datasets = [significant_model_data, model_data]
datanames = ['significant_genes', 'all_genes']

for d, dataset in  enumerate(datasets):
    print('modelled age differences using data from ' + datanames[d])
    all_samples = np.unique(dataset['sample'])
    out_data = dataset[['sample', 'region', 'age', 'PC']].copy()
    correlations = []

    for n, s in enumerate(all_samples):
        # use expression from all other samples as training
        training_data = dataset.loc[(dataset['sample']!=s),:]

        # sample data for testing
        testing_data = dataset.loc[(dataset['sample']==s),:]

        # predicted age differences (with n_boot bootstrapped gene selections)
        predicted_age, correlation = predict_age_difference(training_data, testing_data, pipe, n_boot=n_boot)

        # record output
        out_data.loc[(out_data['sample']==s), 'mean_prediction'] = np.mean(predicted_age, axis=1)
        out_data.loc[(out_data['sample']==s), 'upper_prediction'] = np.percentile(predicted_age, 97.5, axis=1)
        out_data.loc[(out_data['sample']==s), 'lower_prediction'] = np.percentile(predicted_age, 2.5, axis=1)

        correlations.append(correlation)
    
    # get mean [+CI] correlations between PC and (bootstrapped) age differences for each subject
    correlations = np.vstack(correlations)
    out_correlations = pd.DataFrame(all_samples, columns=['sample'])
    out_correlations['age'] = out_correlations['sample'].map(dict(zip(dataset['sample'], dataset['age'])))
    out_correlations['mean_correlation'] = np.mean(correlations, axis=1)
    out_correlations['upper_correlation'] = np.percentile(correlations, 97.5, axis=1)
    out_correlations['lower_correlation'] = np.percentile(correlations, 2.5, axis=1)

    # save out
    outFile_predictions = 'results/age-prediction-' + datanames[d] + '.csv'
    print('see: ', outFile_predictions)
    out_data.to_csv(outFile_predictions, index=None)
    
    outFile_correlations = 'results/PC1-correlations-' + datanames[d] + '.csv'
    print('see: ', outFile_correlations)
    out_correlations.to_csv(outFile_correlations, index=None)
    print('\n  mean correlation: PC1 and age error = %0.2f, p= %0.5f \n' % pearsonr(out_correlations['age'], out_correlations['mean_correlation']))
    print("")

################################################################################################################

### PLOTTING ###################################################################################################
# plots 
print("")
print('calculating correlations between predicted age difference and PC coordinates for each sample...')
for d, dataset in  enumerate(datasets):
    prediction_df = pd.read_csv('results/age-prediction-' + datanames[d] + '.csv')
    correlation_df = pd.read_csv('results/PC1-correlations-' + datanames[d] + '.csv')
    
    sns.lmplot('age', 'mean_prediction', hue='PC', 
               data=prediction_df, height=5,
               fit_reg=True, ci=None, lowess=False, palette='GnBu', truncate=True, legend=False, n_boot=1000, 
               line_kws={'alpha':0.8, 'lw':5}, scatter_kws={'s':40})
    sns.despine(top=False, right=False)
    plt.xlabel('sample age', fontsize=15)
    plt.ylabel('predicted age', fontsize=15)
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
	
    if datanames[d]=='significant_genes':
    	plt.savefig('graphs/Figure4B.pdf')
    	print('see graphs/Figure4B.pdf')
    else:
        plt.savefig('graphs/FigureS7A.pdf')
        print('see graphs/FigureS7A.pdf')

    fig, ax1 = plt.subplots(1,1, figsize=(5,5))

    sns.regplot('age', 'mean_correlation', data=correlation_df, ci=None, color='#65bfcc')
    sns.despine(top=False, right=False)

    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.set_xlabel('sample age', fontsize=15)
    ax1.set_ylabel('correlation', fontsize=15)

    ax1.errorbar(correlation_df['age'], correlation_df['mean_correlation'], correlation_df['upper_correlation'] - correlation_df['mean_correlation'], fmt='none', color='#65bfcc')
    ax1.text(220,0.3, ('r = %0.2f' % (np.corrcoef(correlation_df['age'], correlation_df['mean_correlation'])[0,1])), fontsize=15)
    plt.tight_layout()
    plt.xlim(75, 270)

    if datanames[d]=='significant_genes':
    	plt.savefig('graphs/Figure4C.pdf')
    	print('see graphs/Figure4C.pdf')
    else:
        plt.savefig('graphs/FigureS7B.pdf')
        print('see graphs/FigureS7B.pdf')
    


