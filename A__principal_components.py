#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import os

def main():
    os.makedirs('results/PCA', exist_ok=True)

    ss = StandardScaler()

    # LOAD
    # term group average metrics
    mean_regional_data = pd.read_csv('data/processed_imaging/regional-multimodal-metrics-term-data-clean-zscore-groupaverage.csv', index_col=0)
    # individual term and preterm metrics (cleaned, scaled)
    scaled_term_clean_data = pd.read_csv('data/processed_imaging/regional-multimodal-metrics-term-data-clean-zscore.csv')
    scaled_preterm_clean_data = pd.read_csv('data/processed_imaging/regional-multimodal-metrics-preterm-data-clean-zscore.csv')

    # PCA with term mean regional data
    print("")
    print("performing PCA on mean regional data")
    M = mean_regional_data.T  # rotate to n_regions on metric axes (region x metric)
    M = ss.fit_transform(M)

    # eigendecomposition of cov matrix
    mean_PCs, mean_eigen_vectors, eig_vals = run_PCA(M)

    # results
    metric_vectors = pd.DataFrame([mean_regional_data.index.values, mean_eigen_vectors[:,0]]).T
    metric_vectors.columns = ['metric', 'PC1']
    metric_vectors.to_csv('results/PCA/mean-regional-principal-components-eigenvectors.csv', index=None)
    print('see: results/PCA/mean-regional-principal-components-eigenvectors.csv')

    pc_data = pd.DataFrame(mean_regional_data.columns, columns=['region'])
    pc_data['PC1'] = mean_PCs[:,0]
    pc_data['PC2'] = mean_PCs[:,1]

    pc_data.to_csv('results/PCA/mean-regional-principal-components.csv', index=None)
    print("see: results/PCA/mean-regional-principal-components.csv")

    print("")
    print("projecting individual data onto mean axes..")
    xy_coordinates = []
    for group, dat in zip(['term', 'preterm'], [scaled_term_clean_data, scaled_preterm_clean_data]):
        # project individual data on the mean PC axes and calculate explained variance
        coordinates = calculate_xy(dat, ss, mean_eigen_vectors)
        coordinates['group']=group

        xy_coordinates.append(coordinates)
        coordinates.to_csv('results/PCA/subject-specific-regional-principal-components-{:}.csv'.format(group), index=None)
        print('see: results/PCA/subject-specific-regional-principal-components-{:}.csv'.format(group))

    # concatentate
    all_xy = pd.concat(xy_coordinates, axis=0)

    # get explained variance per subject
    region_average_all_xy=all_xy.groupby('ids').mean()
    region_average_all_xy['group']=all_xy.groupby('ids')['group'].first()
    region_average_all_xy.loc[:,['EV1','age_at_scan','age_at_birth','male','group']].to_csv('results/PCA/principal-components-explained-variance-per-subject.csv')
    print('see: results/PCA/principal-components-explained-variance-per-subject.csv')


#####################################################################
# HELPERS
def run_PCA(data):
    """
    perform PCA via eigendecomposition of covariance matrix
    returns:
    PCs: principal components (data . eig_vecs)
    eigen_vectors, eig_vals: from decomposition of cov matrix
    """
    cov_matrix = np.cov(data, rowvar=False, ddof=1)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    ix = np.argsort(-eig_vals)

    eig_vals = eig_vals[ix]
    eig_vecs= eig_vecs[:,ix]

    eigen_vectors = eig_vecs[:,:]

    # principal components
    PCs = np.dot(data, eigen_vectors)

    return PCs, eigen_vectors, eig_vals


def calculate_xy(data, scaler, eigen_vectors):
    """
    use a set of eigevectors to project new data samples into PC space

    data: dataframe, containing subject level metrics
    scaler: pre-fit StandardScaler with parameters from original data
    eigen_vectors: array, precalculated eigenvectors

    returns:
    xy_coordinates: pandas dataframe, with subject specific PCs and explained variance
    """
    # project individual data onto mean PC space
    xy_coordinates=pd.DataFrame()

    for s in np.unique(data['ids']):
        # select subject specific matrix
        subject_data = data.loc[data['ids']==s,:]
        subject_data = subject_data.sort_values(by=['ids','metric']).loc[:,'A1C':'VLPFC']
        S = scaler.transform(subject_data.T)

        # project onto mean PCs
        masked_arr = np.ma.array(S, mask=np.isnan(S))
        subject_PCs = np.ma.dot(masked_arr, eigen_vectors).data
        explained_variance = np.var(subject_PCs,0)/sum(np.var(subject_PCs,0))

        # x,y coordinates for each subject in PC space
        subject_coordinates = pd.DataFrame([(subject_data.columns.values), subject_PCs[:,0]]).T

        subject_coordinates.columns = ['region', 'PC1']
        subject_coordinates.insert(1, column='EV1', value=[explained_variance[0]]*len(subject_PCs))
        subject_coordinates.insert(0,column='ids',value=[s]*len(subject_PCs))
        subject_coordinates.loc[subject_coordinates.loc[:,'ids']==s, 'age_at_scan'] = data.loc[data.loc[:,'ids']==s,'age_at_scan'].values[0]
        subject_coordinates.loc[subject_coordinates.loc[:,'ids']==s, 'age_at_birth'] = data.loc[data.loc[:,'ids']==s,'age_at_birth'].values[0]
        subject_coordinates.loc[subject_coordinates.loc[:,'ids']==s, 'male'] = data.loc[data.loc[:,'ids']==s,'male'].values[0]

        xy_coordinates = pd.concat([xy_coordinates, subject_coordinates], axis=0)

    xy_coordinates.index=np.arange(len(xy_coordinates))

    return xy_coordinates

if __name__ == '__main__':
    main()
