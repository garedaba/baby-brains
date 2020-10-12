# libraries
library(tidyverse)
library(nlme)
library(mgcv)

# load bulk RPKM data for prenatal samples
normed_rna_data <- read.csv('data/processed_psychencode/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2.csv')

# load  RPKM data for all samples up to 400days
normed_W5_rna_data <- read.csv('data/processed_psychencode/PsychENCODE-W5-bulk-RPKM-data-scRNA-filtered-Winsor-log2.csv')

# load PCA data from imaging
pc_data <- read.csv('results/PCA/mean-regional-principal-components.csv')

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
write.csv(residualised_rna_data, 'data/processed_psychencode/PsychENCODE-prenatal-bulk-RPKM-data-scRNA-filtered-Winsor-log2-residualised.csv', row.names = FALSE)


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
write.csv(model_fit, 'results/gene_correlations/PCA_correlations-KendallTau-residualisedRPKM.csv', row.names = FALSE)


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

cat('calculating expected data predictions for regional model with age....')
trajectories_orig_no_age<- rna_models %>%
  group_by(symbol) %>%
  # set region to the same for all samples
  mutate(data2 = map(data, ~ .x %>% mutate(region = 'A1C'))) %>%
  # get predictions using just sex, RIN and sample intercept (region-specific age effects set to 0, region-specific intercept set to same)
  mutate(predicted = map2(regional_nl_model_result, data2, ~predict(.x, newdata=.y, exclude=exclude_terms, se.fit=TRUE))) %>%
  mutate(fit = map(predicted, ~.x$fit)) %>%
  mutate(se = map(predicted, ~.x$se.fit)) %>%
  mutate(ci = map2(se, 1.96, ~.x * .y))  %>%
  mutate(upper = map2(fit, ci, ~.x + .y)) %>%
  mutate(lower = map2(fit, ci, ~.x - .y)) %>%
  select(symbol, data, fit, upper, lower) %>%
  unnest_legacy() %>%
  ungroup()


trajectories_orig_no_age$residuals <- (trajectories_orig_no_age$log2_rpkm - trajectories_orig_no_age$fit) + trajectories_orig_no_age$mean_rpkm
write.csv(trajectories_orig_no_age, 'results/gene_models/gene-data-corrected.csv', row.names = FALSE)

### MODEL TRAJECTORIES ACROSS FULL GESTATION PERIOD #####################################################################################
# create new dataset spanning full age range
age_seq <- seq(from=80, to=260, length.out=50)
new_x = expand.grid(region=c('A1C','DLPFC','IPC','ITC','M1','MFC','OFC','S1','STC', 'V1','VLPFC'),
                    age=age_seq,
                    RIN=mean(residualised_rna_data$RIN),  # mean RIN across group
                    sex='M',    # arbitrary, will just induce a linear shift up or down for all trajectories
                    sample='HSB159') #  sample is ignored anyway

# predict/model/simulate trajectories across full age span with SE for each region
regional_trajectories<- rna_models %>%
  group_by(symbol) %>%
  mutate(predicted = map(regional_nl_model_result, ~predict(.x, newdata=new_x, exclude=c(s(sample)), se.fit=TRUE))) %>%
  mutate(fit = map(predicted, ~.x$fit)) %>%
  mutate(se = map(predicted, ~.x$se.fit)) %>%
  mutate(ci = map2(se, 1.96, ~.x * .y))  %>%
  mutate(upper = map2(fit, ci, ~.x + .y)) %>%
  mutate(lower = map2(fit, ci, ~.x - .y)) %>%
  select(symbol, fit, upper, lower) %>%
  unnest_legacy() %>%
  ungroup()

# add in age, cluster detail etc
all_new_x <- map_dfr(seq_len(length(unique(regional_trajectories$symbol))), ~new_x)
regional_trajectories <- cbind(all_new_x, regional_trajectories)
cat('saving...\n')
write.csv(regional_trajectories, 'results/gene_models/prenatal-gene-trajectories-by-region.csv', row.names = FALSE)

## W5 DATA ############################################################################################################################################
# repeat with W5 data for genetic age prediction

cat('running W5 models...')
rna_models <- normed_W5_rna_data %>%
  # for each GENE
  group_by(symbol) %>%
  # calculate mean_rpkm
  mutate(mean_rpkm = mean(log2_rpkm)) %>%
  nest()


# fit mixed effects nonlinear models
rna_models <- rna_models %>%
  # for each gene
  group_by(symbol) %>%
  #NONLINEAR with REGION as an additional factor to account for spatial variation across cortex
  mutate(regional_model_result = map(data,
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


cat('calculating expected data predictions for regional model with age....')
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

trajectories_orig_no_age<- rna_models %>%
  group_by(symbol) %>%
  # set region to the same for all samples
  mutate(data2 = map(data, ~ .x %>% mutate(region = 'A1C'))) %>%
  # get predictions using just sex, RIN and sample intercept (region-specific age effects set to 0, region-specific intercept set to same)
  mutate(predicted = map2(regional_model_result, data2, ~predict(.x, newdata=.y, exclude=exclude_terms, se.fit=TRUE))) %>%
  mutate(fit = map(predicted, ~.x$fit)) %>%
  mutate(se = map(predicted, ~.x$se.fit)) %>%
  mutate(ci = map2(se, 1.96, ~.x * .y))  %>%
  mutate(upper = map2(fit, ci, ~.x + .y)) %>%
  mutate(lower = map2(fit, ci, ~.x - .y)) %>%
  select(symbol, data, fit, upper, lower) %>%
  unnest_legacy() %>%
  ungroup()

# residuals contain all variance not explained by sex, RIN and sample (i.e.: variation in expression due to region and age)
trajectories_orig_no_age$residuals <- (trajectories_orig_no_age$log2_rpkm - trajectories_orig_no_age$fit) + trajectories_orig_no_age$mean_rpkm
write.csv(trajectories_orig_no_age, 'results/gene_models/W5-gene-data-corrected.csv', row.names = FALSE)
