library(tidyverse)

# load RPKM models over prenatal period
trajectory_data <- read.csv('results/gene_models/prenatal-gene-trajectories-by-region.csv')

# load group mean differences
group_differences <- read.csv('results/metrics/metric_differences.csv')
# keep just T1/T2
group_differences <- group_differences[group_differences$metric=='myelin',]

# map differences to each region
trajectory_data$difference <- plyr::mapvalues(trajectory_data$region, group_differences$region, group_differences$value)
trajectory_data$difference <- as.numeric(levels(trajectory_data$difference))

# keep older samples in preterm period
age_limit = 160
trajectory_data <- trajectory_data[trajectory_data$age>age_limit,]
trajectory_data$age_windows <- cut(trajectory_data$age, 10, labels=FALSE)

model_fit <- trajectory_data %>%
  # for each GENE
  group_by(symbol, age_windows) %>%
    nest() %>%
    # fit Kendall's tau
    # PC1
    mutate(corr = map(data, ~cor.test(~ fit + difference, data=., method="kendall", exact=FALSE))) %>%
    mutate(corr_result = map(corr, broom::tidy)) %>%
    mutate(corr_tau = map(corr_result, select, estimate)) %>%
    mutate(corr_zstat = map(corr_result, select, statistic)) %>%
    mutate(corr_pval = map(corr_result, select, p.value)) %>%
    # results
    select('symbol', 'age_windows', 'corr_tau', 'corr_pval') %>%
    unnest_legacy() %>%
    ungroup()

colnames(model_fit) <- c('symbol', 'age','tau', 'pval')
cat('saving...')
write.csv(model_fit, 'results/gene_correlations/windowed_correlations.csv', row.names = FALSE)
