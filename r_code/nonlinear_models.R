# libraries
library(tidyverse)
library(nlme)
library(mgcv)

# load bulk RPKM data
normed_rna_data <- read.csv('data/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2.csv')

# load PCA data from imaging
pc_data <- read.csv('results/mean-regional-principal-components.csv')

# get PCA coordinates for each region
for (r in c('A1C','IPC', 'DLPFC','ITC',"OFC", "VLPFC", "MFC", 'STC', 'M1', 'S1', 'V1')){
  normed_rna_data[normed_rna_data$region==r, 'pc1'] <- pc_data[pc_data$region==r,'PC1']
}

## MODELLING ####################################################################################################################################
# used mixed effect (non)linear model to account for variation due to sample (random effect), main effects of age, RIN and sex
## first calculate mean rpkm for each gene
cat('running models...')
rna_models <- normed_rna_data %>%
  # for each GENE
  group_by(symbol) %>%
  # calculate mean_rpkm
  mutate(mean_rpkm = mean(log2_rpkm)) %>%
  nest()


# fit mixed effects nonlinear models
rna_models <- rna_models %>%
  # for each gene
  group_by(symbol) %>%
  #NONLINEAR
  mutate(nl_model_result = map(data,
                               ~gam(log2_rpkm ~
                                    # random intercept of sample
                                    1 + s(sample, bs='re')
                                    # main effects: s(age), with RNA integrity and sex as confounders
                                    # age fit with natural cubic spline with 4 knots spaced evenly across age span
                                    + s(age, k=4, bs='cs', fx=TRUE)
                                    + RIN + sex,
                                    data =.))) %>%

  #NONLINEAR with REGION as an additional factor to account for spatial variation across cortex
  mutate(regional_nl_model_result = map(data,
                                    ~gam(log2_rpkm ~
                                         # random intercept of sample
                                         1 + s(sample, bs='re')
                                         # smooth of age, varying for each region, fixed knots
                                         + s(age, k=4, by=as.factor(region), bs='cs', fx=TRUE)
                                         # main effect of region to keep region means
                                         + as.factor(region)
                                         # sex and RIN as before
                                         + sex + RIN,
                                         data=.)))

# collect residuals for spatial correlation analysis
rna_models <- rna_models %>%
  group_by(symbol) %>%
  # collect residuals (i.e.; corrected RPKM data) for linear and nonlinear models
  mutate(residuals_nl = map(nl_model_result, residuals))


# model residuals
residualised_rna_data <- rna_models %>%
  select('symbol', 'residuals_nl', 'data') %>%
  unnest_legacy() %>%
  # add mean expression back to residuals
  mutate(mean_nl_residuals = map2(residuals_nl, mean_rpkm, ~.x + .y)) %>%
  unnest_legacy() %>%
  select(-c(residuals_nl)) %>%
  ungroup()

cat('saving residualised_rna_data...')
write.csv(residualised_rna_data, 'data/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2-residualised.csv', row.names = FALSE)


### CORRELATION ANALYSIS ################################################################################################################
cat('')
cat('running correlations...')
model_fit <- residualised_rna_data %>%

  # for each GENE
  group_by(symbol) %>%
  nest() %>%

  # calculate Kendall's tau between residualised log2RPKM (corrected for age, sex, etc) and principal component order
  mutate(pc1_corr = map(data, ~cor.test(~ mean_nl_residuals + pc1, data=., method="kendall"))) %>%
  mutate(pc1_corr_result = map(pc1_corr, broom::tidy)) %>%
  mutate(pc1_corr_tau = map(pc1_corr_result, select, estimate)) %>%
  mutate(pc1_corr_zstat = map(pc1_corr_result, select, statistic)) %>%
  mutate(pc1_corr_pval = map(pc1_corr_result, select, p.value)) %>%

  # results
  select('symbol', 'pc1_corr_tau', 'pc1_corr_pval') %>%
  unnest_legacy() %>%
  ungroup()

# save results
colnames(model_fit) <- c('symbol', 'PC1_tau', 'PC1_pval')
cat('saving...')
write.csv(model_fit, 'results/PCA_correlations-KendallTau-residualisedRPKM.csv', row.names = FALSE)

### CALCULATE CORRECTED SAMPLE TRAJECTORIES  #####################################################################
# residuals of this model = gene expression corrected for variance due to sex, RIN, and specimen ID while retaining
# (region-specific) variance due to age

exclude_terms <- c('s(age):as.factor(region)A1C',
                   's(age):as.factor(region)DLPFC',
                   's(age):as.factor(region)IPC',
                   's(age):as.factor(region)ITC',
                   's(age):as.factor(region)M1',
                   's(age):as.factor(region)MFC',
                   's(age):as.factor(region)OFC',
                   's(age):as.factor(region)S1',
                   's(age):as.factor(region)STC',
                   's(age):as.factor(region)V1',
                   's(age):as.factor(region)VLPFC')

cat('calculating expected data predictions for cluster model (cluster_model) excluding age....')
trajectories_orig_no_age<- rna_models %>%
  group_by(symbol) %>%
  # set region to the same for all samples
  mutate(data2 = map(data, ~ .x %>% mutate(region = 'A1C'))) %>%
  # get predictions using just sex, RIN and sample intercept (region-specific age effects set to 0, region-specific intercept set to same)
  mutate(predicted = map2(regional_model_result, data2, ~predict(.x, newdata=.y, exclude=exclude_terms, se.fit=TRUE))) %>%

  mutate(predicted = map(regional_nl_model_result, ~predict(.x, exclude=exclude_terms, se.fit=TRUE))) %>%
  mutate(fit = map(predicted, ~.x$fit)) %>%
  mutate(se = map(predicted, ~.x$se.fit)) %>%
  mutate(ci = map2(se, 1.96, ~.x * .y))  %>%
  mutate(upper = map2(fit, ci, ~.x + .y)) %>%
  mutate(lower = map2(fit, ci, ~.x - .y)) %>%
  select(symbol, data, fit, upper, lower) %>%
  unnest_legacy() %>%
  ungroup()


trajectories_orig_no_age$residuals <- (trajectories_orig_no_age$log2_rpkm - trajectories_orig_no_age$fit) + trajectories_orig_no_age$mean_rpkm
write.csv(trajectories_orig_no_age, 'results/gene-data-corrected.csv', row.names = FALSE)
