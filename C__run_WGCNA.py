import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

import subprocess, os

def main():
    os.makedirs('results/wgcna', exist_ok=True)
    ### RUN WGCNA ####################################################################################
    print("running WGCNA in R")
    print("")
    retcode = subprocess.call(['/usr/bin/Rscript','./r_code/WGCNA_analysis.R'])
    print("")
    if retcode==1:
        print("something went wrong...")

    ##################################################################################################

    ### CONVERT OUTPUT TO NETWORKS FOR VISUALISATION #################################################
    for pos_or_neg in ['positive', 'negative']:
        print("")
        print('extracting module networks for '+pos_or_neg+' gene associations..')
        # load data
        data = pd.read_csv('results/wgcna/WGCNA_'+pos_or_neg+'_TOM_matrix.csv', index_col=0)
        node_info = pd.read_csv('results/wgcna/WGCNA_'+pos_or_neg+'_node_modules.csv', index_col=None)

        # identify unassigned genes
        unassigned = node_info['gene_names'].loc[node_info['moduleid']==0]
        print(str(len(unassigned)) + ' genes removed: ')
        print(unassigned.values)

        # zero_diagnonal
        np.fill_diagonal(data.values, 0)

        # order data by module id and drop unassigned genes
        order_data = (data.iloc[node_info['module_order']-1,node_info['module_order']-1])
        order_data.drop(unassigned,0, inplace=True)
        order_data.drop(unassigned,1, inplace=True)

        ordered_node_info = node_info.iloc[node_info['module_order'].values-1].reset_index()
        ordered_node_info.drop(np.where(ordered_node_info.gene_names.isin(unassigned))[0], inplace=True)

        # save order data for later plotting
        order_data.to_csv('results/wgcna/WGCNA_expression_matrix-' + pos_or_neg + '.csv', index=None)
        print('see: results/wgcna/WGCNA_expression_matrix-' + pos_or_neg + '.csv')

        # output module specific networks
        print("")
        print(str(max(ordered_node_info.moduleid)) + ' modules found')
        for i in np.unique(ordered_node_info.moduleid):
            genes_in_mod = order_data.index[np.where(ordered_node_info.moduleid==i)]
            mod_data = order_data.loc[genes_in_mod,:].loc[:,genes_in_mod]

            mod_info = ordered_node_info.loc[ordered_node_info.moduleid==i,:]
            graph = convert_to_network(mod_data, mod_info)

            nx.write_graphml(graph, 'results/wgcna/WCGNA_'+pos_or_neg+'_module_' + str(i) + '.graphml')
            print('see: results/wgcna/WGCNA_'+pos_or_neg+'_module_' + str(i) + '.graphml')
    print("")
    print('see results/wgcna/WGCNA*graphml files for visualisation of module networks in Cytoscape (or your choice of software)')

### HELPERS #####################################################################
def convert_to_network(matrix, matrix_info, prc=50):

    thresholded_matrix = matrix * (matrix>np.percentile(matrix.values, prc))

    # make networkx object
    g = nx.to_networkx_graph(thresholded_matrix.values)

    # relabel nodes
    node_dict = dict(zip(np.arange(len(matrix.index)), matrix.index))
    g = nx.relabel_nodes(g, node_dict)

    # add atrributes - module ID, strength
    strength_dict = dict(zip(matrix_info['gene_names'], matrix_info['node_strength']))
    nx.set_node_attributes(g, strength_dict, 'strength')

    return g

if __name__ == '__main__':
    main()
