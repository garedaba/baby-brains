import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from scipy.stats import pearsonr, zscore

from tqdm import tqdm
from joblib import Parallel, delayed

import os

# number of bootstraps
n_boot = 100

ss = StandardScaler()
kr = SVR(kernel='linear', C=10)
pipe = Pipeline([('scaler',ss), ('regressor', kr)])

def main(permute=False):
    os.makedirs('results/age_predictions', exist_ok=True)

    ### LOAD DATA ##################################################################################################
    # load sample, sex, RIN corrected data for all samples up to 400 days
    bulk_data = pd.read_csv('results/gene_models/W5-gene-data-corrected.csv')
    significant_genes = pd.read_csv('results/gene_correlations/PCA_correlations-KendallTau-PC-significant_genes-p0.05.csv')['symbol']

    # load PC data
    pca_data = pd.read_csv('results/PCA/mean-regional-principal-components.csv')
    pca_dict = dict(zip(pca_data['region'], pca_data['PC1']))

    # pivot table
    model_data = pivot_bulk_data(bulk_data, values='residuals')  # sample data, all genes
    model_data.insert(3, 'PC', model_data['region'].map(pca_dict))

    # select significant genes
    significant_model_data = pd.concat((model_data[['sample','region','age', 'PC']], model_data[significant_genes]), axis=1)

    # validation data
    braincloud_data = pd.read_csv('data/validation/BrainCloud-ALL-bulk-expression-data-scRNA-filtered.csv')
    # limit to sames age range
    braincloud_data = braincloud_data[braincloud_data.age<400]

    n_braincloud_samples = len(braincloud_data)

    braincloud_metadata = braincloud_data[['sample','age']]
    braincloud_data_genes = braincloud_data.drop(['sample','age'], axis='columns')
    gene_ids = list(braincloud_data_genes.columns)

    ### MODELLING 1##################################################################################################
    # use mean cortical data to predict regional age based on gene expression for all samples 8pcw->4months
    # calculate correlation between PC1 and predicted age errors
    datasets = [significant_model_data, model_data]
    datanames = ['significant_genes', 'all_genes']

    for d, dataset in  enumerate(datasets):
        print('')
        print('\nmodelling using {:}'.format(datanames[d]))
        all_samples = np.unique(dataset['sample'])
        out_data = dataset[['sample', 'region', 'age', 'PC']].copy()
        bs_correlations = []

        sample_predictions = np.zeros((len(out_data), n_boot))

        # cortical average expression data
        mean_regional_data = dataset.groupby(['sample', 'age']).mean().reset_index()

        # LOO CV
        for b in tqdm(np.arange(n_boot), desc='bootstraps'):
            n_genes = dataset.drop(['region','sample','age','PC'], axis='columns').shape[1]

            # select bootstrapped sample of genes for model
            idx = np.random.choice(np.arange(n_genes), size=n_genes, replace=True)

            correlations = []
            # for each sample
            for n, s in enumerate(tqdm(all_samples, desc='subjects')):
                # use expression from all other samples as training
                training_data = mean_regional_data.loc[(dataset['sample']!=s),:].drop(['sample','age','PC'], axis='columns')
                training_age = mean_regional_data.loc[(dataset['sample']!=s),'age']
                training_samples = mean_regional_data.loc[(dataset['sample']!=s),'sample']
                # regional data for testing
                testing_data = dataset.loc[(dataset['sample']==s),:].drop(['sample','age','region','PC'], axis='columns')
                testing_PC = dataset.loc[(dataset['sample']==s),'PC']

                # fit, predict age using SVR, bootstrap gene selections n_boot times
                predictions = bootstrap_predictions(training_data, training_age, testing_data, pipe, boot_idx=idx, permute_idx=None)
                # age predictions for each region based on bootstrapped genes
                sample_predictions[out_data['sample']==s,b] = predictions
                # correlation between PC and age predictions (equiv. to age predicted error as all regions are the same 'age')
                correlations.append(pearsonr(testing_PC, predictions)[0])

            # keep all correlations across bootstrap samples for later
            bs_correlations.append(np.stack(correlations))

        # summarise predictions for output and plotting
        out_data.loc[:, 'mean_prediction'] = np.mean(sample_predictions, axis=1)
        out_data.loc[:, 'lower_prediction'], out_data.loc[:, 'upper_prediction'] = np.percentile(sample_predictions, [2.5,97.5], axis=1)
        # highlight prenatal samples
        out_data.insert(3,'prenatal',((out_data.age>80) & (out_data.age < 300)).astype(int))

        # stack up correlations for all bootstrap samples
        all_correlations = pd.DataFrame(np.vstack(bs_correlations).T)
        out_correlations = pd.DataFrame(all_samples, columns=['sample'])
        out_correlations['age'] = out_correlations['sample'].map(dict(zip(dataset['sample'], dataset['age'])))

        # add summary data to output file
        out_correlations['mean_correlation'] = np.mean(all_correlations, axis=1)
        out_correlations['lower_correlation'], out_correlations['upper_correlation'] = np.percentile(all_correlations, [2.5,97.5], axis=1)
        # prenatal samples
        out_correlations.insert(2,'prenatal',((out_correlations.age>80) & (out_correlations.age < 300)).astype(int))


        # for each bootstrapped gene sample, fit a model to predict PC-correlation from age for all prenatal samples
        # this ensures that, for each bootstrap sample, individual correlation estimates are grouped together, as they are estimated using the same genes
        correlation_data = pd.concat((out_correlations, all_correlations), axis=1)
        prenatal_model_fits, prenatal_r2, x_range = calculate_model_fits(correlation_data[correlation_data['prenatal']==1])

        print("")
        print('mean model fit: PRENATAL samples [y~x] R^2 = {:.2f}'.format(np.mean(prenatal_r2)))


        ### SAVE OUT ################################################################################################
        # REGIONAL AGE PREDICTIONS
        outFile_predictions = 'results/age_predictions/age-prediction-' + datanames[d] + '.csv'
        print('see: ', outFile_predictions)
        out_data.to_csv(outFile_predictions, index=None)

        # PC1 CORRELATIONS
        outFile_correlations = 'results/age_predictions/PC1-correlations-' + datanames[d] + '.csv'
        print('see: ', outFile_correlations)
        out_correlations.to_csv(outFile_correlations, index=None)

        # PC1 CORRELATIONS x AGE MODEL FITS
        model_fit_df = pd.DataFrame(x_range, columns=['age'])
        model_fit_df['prenatal_mean_fit'] = np.mean(prenatal_model_fits, axis=1)
        model_fit_df['prenatal_lower_fit'], model_fit_df['prenatal_upper_fit'] = np.percentile(prenatal_model_fits, [2.5,97.5], axis=1)

        print('see: results/age_predictions/PC1-correlations-' + datanames[d] + '-model-fits.csv')
        model_fit_df.to_csv('results/age_predictions/PC1-correlations-' + datanames[d] + '-model-fits.csv', index=None)

        # EXPLAINED VARIANCE FOR EACH MODEL - for comparison to permuted r2 values
        model_results = pd.DataFrame(['prenatal', np.mean(prenatal_r2),  np.percentile(prenatal_r2, 2.5), np.percentile(prenatal_r2, 97.5)]).T
        model_results.columns=['samples', 'r2', 'upper', 'lower']
        print('see: results/age_predictions/PC1-correlations-' + datanames[d] + '-model-results.csv')
        model_results.to_csv('results/age_predictions/PC1-correlations-' + datanames[d] + '-model-results.csv', index=None)
        print("")


        ### VALIDATION ################################################################################################
        print('')
        print('validating model in BrainCloud')
        print('number of BrainCloud samples: {:}'.format(n_braincloud_samples))
        print('')
        print('modelled age differences using data from ' + datanames[d])

        mean_training_data = dataset.groupby(['sample', 'age']).mean().reset_index()
        # extract gene columns
        mean_training_data_genes = mean_training_data.drop(['sample','age'], axis='columns')
        # extract age columns
        mean_training_data_age = mean_training_data['age']

        # keep only genes in both cohorts
        training_gene_ids = list(mean_training_data_genes.columns)
        joint_ids = sorted(list(set(gene_ids) & set(training_gene_ids)))
        print('total number of genes in both datasets: {:}'.format(len(joint_ids)))

        shared_training_data_genes = mean_training_data_genes.loc[:,joint_ids]
        shared_braincloud_data_genes = braincloud_data_genes.loc[:,joint_ids]

        # make sure the datasets match
        if sum(shared_braincloud_data_genes.columns == shared_training_data_genes.columns) != len(joint_ids):
            raise ValueError('gene order does not match - please CHECK!')

        # zscore to bring both datasets to same scale
        shared_training_data_genes = shared_training_data_genes.apply(zscore)
        shared_braincloud_data_genes = shared_braincloud_data_genes.apply(zscore)

        # fit model
        predictions = np.zeros((len(braincloud_data_genes), n_boot))
        for b in tqdm(np.arange(n_boot)):
            idx = np.random.choice(np.arange(len(joint_ids)), size=len(joint_ids), replace=True)
            predictions[:,b] = bootstrap_predictions(shared_training_data_genes, mean_training_data_age, shared_braincloud_data_genes, pipe, boot_idx=idx, permute_idx=None)

        # collate and save out
        out_data = braincloud_metadata.copy()
        out_data['mean_prediction'] = np.mean(predictions, axis=1)
        out_data['upper_prediction'] = np.percentile(predictions, 97.5, axis=1)
        out_data['lower_prediction'] = np.percentile(predictions, 2.5, axis=1)
        out_data.insert(2,'prenatal',((out_data.age>80) & (out_data.age < 300)).astype(int))
        print('see: results/age_predictions/age-prediction-BrainCloud-' + datanames[d] + '.csv')
        out_data.to_csv('results/age_predictions/age-prediction-BrainCloud-' + datanames[d] + '.csv', index=None)
        print('')



#### HELPERS ###################################################################
def calculate_model_fits(correlation_data):
    """
    model change in PC1 correlation over age using prenatal samples.
    models:  y~b.x for prenatal samples

    correlation_data: dataframe, containing sample 'age', and n_boot correlations (numbered 0 to n_boot)

    returns:
    prenatal_model_fits: f(x) over full age range for each bootstrap sample (fit using prenatal data only)
    prenatal_r2: variance explained by model for each bootstrap sample (for prenatal samples only)
    fx: x for f(x)
    """
    n_boot = correlation_data.columns[-1] + 1

    prenatal_model_fits = np.zeros((100, n_boot))
    prenatal_r2 = np.zeros(n_boot)

    # calculate model fits across bootstrap samples
    print('calculating model fits')
    for i in tqdm(np.arange(n_boot), desc='model fits'):
        # prenatal data: y ~ x
        fx, prenatal_model_fits[:,i], prenatal_r2[i] = model_expression(correlation_data['age'].values.reshape(-1,1), correlation_data[i].values.reshape(-1,1), log=False)

    return prenatal_model_fits, prenatal_r2, fx


def pivot_bulk_data(data, values='log2_rpkm'):
    """
    take long format expression data and pivot to a wider (subject x region) x gene dataframe

    data: long format dataframe (gene x region x subject) x rpkm
    values: metric of interest to create a (region x subject) x gene matrix of metrics

    returns:
    table_data: wide format dataframe
    """

    # select data
    selected_data = data.loc[:,['symbol', 'sample', 'region', 'age', values]]
    # pivot
    table_data = selected_data.pivot_table(index=['sample', 'region', 'age'], columns='symbol', values=values)
    table_data = table_data.reset_index()

    return table_data


def bootstrap_predictions(train_x, train_y, test_x, model, boot_idx=None, permute_idx=None):
    """
    fits a model to training data and predicts output using test data. features are selected with boot_idx.
    Optional permutation of rows for significant testing.

    train_x, test_x: dataframes containing train and test data
    train_y: target value for model fitting
    model: sklearn regressor object, or Pipeline
    boot_idx: features to use for model
    permute_idx: permutation index to shuffle training data

    returns:
    pred: bootstrapped predictions of test data
    """

    # fit model with bootstrapped genes
    if boot_idx is None:
        if permute_idx is None:
            model.fit(train_x, train_y)
        else:
            model.fit(train_x.iloc[permute_idx,:], train_y)

        pred = model.predict(test_x)

    else:
        if permute_idx is None:
            model.fit(train_x.iloc[:,boot_idx], train_y)
            pred = model.predict(test_x.iloc[:,boot_idx])
        else:
            model.fit(train_x.iloc[permute_idx,boot_idx], train_y)
            pred = model.predict(test_x.iloc[:,boot_idx])

    return pred


def model_expression(x, y, log=False):
    """
    fit a linear/loglinear model to x to predict y and return model fit

    x, y: data to fit
    log: apply log transform to x first?

    returns:
    fit_x, fit_y: model fit values over range of x
    r2: model fit
    """
    lm = LinearRegression()

    fit_x = np.linspace(50, 400, num=100).reshape(-1,1)

    if log:
        lm.fit(np.log(x), y)
        fit_y = lm.predict(np.log(fit_x))
        r2 = r2_score(y, lm.predict(np.log(x)))
    else:
        lm.fit(x, y)
        fit_y = lm.predict(fit_x)
        r2 = r2_score(y, lm.predict(x))

    return fit_x.reshape(-1), fit_y.reshape(-1), r2


if __name__ == '__main__':
    print('running models')
    main(permute=False)
