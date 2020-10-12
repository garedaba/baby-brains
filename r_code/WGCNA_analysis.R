library(tidyverse)
library(WGCNA)
library(RColorBrewer)
library(vegan)
options(stringsAsFactors = FALSE);

for (pos_or_neg in c('positive', 'negative')){

  # load residualised RPKM data (5287 genes)
  all_data <- read.csv('data/processed_psychencode/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2-residualised.csv')

  # pivot to wide format
  expr_data <- select(all_data, c(symbol, sample, region, mean_nl_residuals))  %>%
                unite(sample_region, c(sample, region)) %>%
                spread(sample_region, mean_nl_residuals)

  # transpose to sample x gene
  expr_data <- expr_data %>%
                gather(key = sample, value = value, 2:ncol(expr_data)) %>%
                spread_(key = names(expr_data)[1],value = 'value')

  # keep only genes-of-interest
  # significant genes
  sig_genes <- read.csv('results/gene_correlations/PCA_correlations-KendallTau-PC-significant_genes-p0.05.csv')
  if (pos_or_neg=='positive'){
    sig_genes <- sig_genes[sig_genes$PC1_tau>0,]$symbol
    } else {
    sig_genes <- sig_genes[sig_genes$PC1_tau<0,]$symbol
    }

  # keep expression data only those genes positively associated with PC1
  expr_data <- expr_data[,(names(expr_data) %in% sig_genes)]

  # The code is based on the WCGNA tutorial: https://horvath.genetics.ucla.edu/html/CoexpressionNetwork/Rpackages/WGCNA/Tutorials/Consensus-NetworkConstruction-auto.pdf
  # default values have been used for most parts
  # set up for wcgna
  gene_names <- colnames(expr_data)
  # just expression data
  expr_data <-fixDataStructure(expr_data, verbose = TRUE)

  # Choose a set of soft-thresholding powers
  powers = c(seq(4,10,by=1), seq(12,20, by=2));
  # Initialize a list to hold the results of scale-free analysis
  powerTables = vector(mode = "list", length = 1);
  # Call the network topology analysis function for each set in turn
  powerTables[[1]] = list(data = pickSoftThreshold(expr_data[[1]]$data, powerVector=powers,
                                                       verbose = 2)[[2]]);
  # results:
  plotData <- powerTables[[1]]$data %>% select(Power, SFT.R.sq)
  # best choice is based on max SFT model fit...
  powerChoice=plotData[which.max(plotData[,2]),1]

  # calculate adjacency matrix
  nGenes =  ncol(expr_data[[1]]$data)
  adjacencies = array(0, dim = c(1, nGenes, nGenes))

  # Calculate adjacencies in each data set
  adjacencies[1, , ] = abs(cor(expr_data[[1]]$data, use = "p"))^powerChoice;

  # convert to TOM
  TOM = array(0, dim = c(1, nGenes, nGenes));
  # Calculate TOMs in each data set
  TOM[1, , ] = TOMsimilarity(adjacencies[1, , ]);

  # Clustering
  TOMmat <- TOM[1, , ]
  # dendrogram
  consTree = stats::hclust(as.dist(1-TOMmat), method = "average");
  # minimum module size relatively high:
  minModuleSize = 5;
  # Module identification using dynamic tree cut:
  modLabels = cutreeDynamic(dendro = consTree, distM = 1-TOMmat,
                                 deepSplit = 2, cutHeight = 0.995,
                                 minClusterSize = minModuleSize,
                                  pamRespectsDendro = TRUE );
  modColors = labels2colors(modLabels, colorSeq = brewer.pal(n = 8, name = "Accent"))
  modColors[modColors=='grey']='grey98'

  # '0' label indicates that genes could not be assigned to a module
  table(modLabels)

  if (max(modLabels)>3){
    # Combine similar modules based on eigengene expression
    # module eigengenes
    unmergedMEs = multiSetMEs(expr_data, colors = NULL, universalColors = modColors, excludeGrey = TRUE)
    # dissimilarity ofmodule eigengenes
    consMEDiss = consensusMEDissimilarity(unmergedMEs);
    # cluster modules
    consMETree = hclust(as.dist(consMEDiss), method = "average");

    # merge modules with similar eigengene expression values (auto cutoff of .25)
    merge = mergeCloseModules(expr_data, modLabels, cutHeight = 0.25, verbose = 3)

    # rename with merged clusters
    # Numeric module labels
    modLabels = merge$colors;
    # Convert labels to colors
    modColors =  labels2colors(modLabels, colorSeq = brewer.pal(n = 8, name = "Accent"))
    modColors[modColors=='grey']='grey98'
    # Eigengenes of the new merged modules:
    consMEs = merge$newMEs;
    }

  # save TOM info
  module_order <- consTree$order
  moduleid <- modLabels
  node_strength <- rowSums(TOMmat)
  node_info <- cbind(gene_names, moduleid, module_order, node_strength)
  outfile <- paste('results/wgcna/WGCNA', pos_or_neg, 'node_modules.csv', sep='_')
  write.csv(node_info, outfile, quote = FALSE)

  # save TOM matrix
  tom_matrix <- as.matrix(TOMmat)
  rownames(tom_matrix) <- gene_names
  colnames(tom_matrix) <- gene_names
  outfile <- paste('results/wgcna/WGCNA', pos_or_neg, 'TOM_matrix.csv', sep='_')
  write.csv(tom_matrix, outfile, quote = FALSE)

}
