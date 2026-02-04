"""
IST Causal econometrics project

Objective: examine the IST data (one of the largest clinical trials ever conducted) 
to investigate the effect of aspirin on stroke outcomes and explore treatment heterogeneity,
particularly focusing on survival and disability at six months.
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from mcf import ModifiedCausalForest

# custom functions (OLS and statistical tests are manually coded)
from src import functions as f

from src.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, IST_DATA_FILE, FIGURES_DIR, RANDOM_SEED

# Load the data
data = pd.read_csv(RAW_DATA_DIR / IST_DATA_FILE)

########################################
# DATA CLEANING
########################################

# check missing values
missing = f.nacount(data)
print(missing, '\n')

# remove obs with nas
data = data.dropna()

# double check the missing values
missing_clean = f.nacount(data)
print(missing_clean, '\n')

# issue: DASP14 should only take values Y/N
print("Unique values of DASP14:", data['DASP14'].unique(), "\n")

# remove U observations and substitute y/n with Y/N
data = data[data['DASP14'] != 'U']
data['DASP14'] = data['DASP14'].str.replace('y', 'Y')
data['DASP14'] = data['DASP14'].str.replace('n', 'N')
print("Post-cleaning unique values of DASP14:", data['DASP14'].unique(), "\n") # double check

# convert binary vars into dummies
for col in data.columns:
    levels = np.sort(np.array(data[col].unique()))
    
    if len(levels) == 2:
        if np.array_equal(levels, np.array(['N', 'Y'])) :
            data[col] = np.where(data[col] == 'Y', 1, 0) # dummy Y/N to 1 if Y 0 if N    
        elif np.array_equal(levels, np.array(['F', 'M'])):
            data[col] = np.where(data[col] == 'M', 1, 0) # dummy M/F to 1 if M, 0 if F
        else:
            pass

# recode RCONSC levels into numerical values
RCONSC_map = {'F': 2, 'D': 1, 'U': 0}
data['RCONSC'] = data['RCONSC'].map(RCONSC_map)

# recode COUNTRY
country_dummies = pd.get_dummies(data['COUNTRY'], prefix='country', dtype=float)

# remove the last country (USA) to avoid multicollinearity issues  
country_dummies = country_dummies.iloc[:, :-1]

# drop the old coutry var and oncatenate the new dummies
data = data.drop('COUNTRY', axis = 1)
data = pd.concat([data, country_dummies], axis=1)

# recode STYPE
stype_dummies = pd.get_dummies(data['STYPE'], prefix='STYPE', dtype=float)

# drop the last STYPE dummy to avoid multicollinearity
stype_dummies = stype_dummies.iloc[:, :-1]

# drop the old STYPE var and concatenate the new dummies
data = data.drop('STYPE', axis = 1)
data = pd.concat([data, stype_dummies], axis=1)

# recode Yes/No/Cannot say variables
ync_vars = ['RDEF1', 'RDEF2', 'RDEF3', 'RDEF4',
            'RDEF5', 'RDEF6', 'RDEF7','RDEF8']

# use Ber(p) outcomes to handle "Cannot say" values with p mirroring the empirical
# probability of observing Y (iflate std errors, but does not bias results)
# estimate empirical probabilities of Y
recode_prob = pd.DataFrame([], index=ync_vars, columns = ['p_yes'])

for var in ync_vars:
    fullinfo = data[data[var] != 'C'][var] # extract the datapoints with clear Y/N
    recode_prob.loc[var, 'p_yes'] = sum(fullinfo == 'Y')/fullinfo.shape[0] # compute the probability of Y

# substitute randomly selected Y/N in place of Cannot say
for target_col in ync_vars:
    # boolean mask of rows to replace
    mask = data[target_col] == 'C'
    
    # draw bernoulli trials and plug Y/N accordingly
    draws = np.random.binomial(1, recode_prob.loc[target_col, 'p_yes'], size=mask.sum())
    data.loc[mask, target_col] = np.where(draws == 1, 'Y', 'N')

# Convert Y/N into dummies and double check unique values
for col in ync_vars:
    data.loc[:,col] = np.where(data.loc[:,col] == 'Y', 1, 0)    
    print(col, 'unique values: ', data[col].unique(), '\n')
    
# analyze outliers
continuous = ['AGE', 'RSBP', 'RDELAY']

# boxplot continuous variables
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax, col in zip(axes, continuous):
    ax.boxplot(data[col])
    ax.set_xlabel(col)
    ax.set_ylabel('Value')

plt.suptitle('Pre-cleaning boxplots')
plt.savefig(FIGURES_DIR / 'pre_cleaning_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# remove patients registered as 120 years old or older
data = data.loc[data['AGE']<120, :]

# boxplot cleaned continuous variables
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax, col in zip(axes, continuous):
    ax.boxplot(data[col])
    ax.set_xlabel(col)
    ax.set_ylabel('Value')

plt.suptitle('Post-cleaning boxplots')
plt.savefig(FIGURES_DIR / 'post_cleaning_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# compute mean and std of all cols by treatment status
mean_sts = f.mean_std(data, 'RXASP')
print(mean_sts, '\n')

########################################
# AVG TREATMENT EFFECT BY MEAN DIFFERENCE
########################################
# call function for mean comparison
meandiff = f.meandiff(data, treatment = 'RXASP', var = 'FDEAD')
print(meandiff, '\n')

########################################
# ATE ESTIMATE WITH OLS
########################################
# Define the covariates to use in OLS regression
OLS_covs = ['RXASP', 'DASP14', 'AGE', 'RSBP', 'RDELAY', 
              'SEX', 'RCT', 'RVISINF', 'RATRIAL']

# Estimate ATE with OLS
OLS_estimates = f.ols(OLS_covs, 'FDEAD', data, intercept=True)
print(OLS_estimates, '\n')

########################################
# PROPENSITY SCORES
########################################
# specify the covariates to be included in the logit
logit_covs = ['AGE', 'RSBP', 'RDELAY', 
              'SEX', 'RCT', 'RVISINF', 'RATRIAL']

# compute the propensity scores
pscores = f.logit_pscores('RXASP', logit_covs, data, intercept=True)
pscores = pd.DataFrame(pscores)

# compute mean variance, max and min of pscores
pscores_data = f.summary_stats(pscores)
print(pscores_data, '\n')

# Alternative approach: use Lasso for covariate selection
data_lasso = data.drop(['FDEAD', 'DASP14'], axis=1)
logit_covs_lasso = f.lasso_selection(data_lasso, 'RXASP')

if len(logit_covs_lasso)>0:
    # compute the propensity scores
    pscores_lasso = f.logit_pscores('RXASP', logit_covs_lasso, data, intercept=True)
    pscores_lasso = pd.DataFrame(pscores_lasso)
    
    # compute mean variance, max and min of pscores
    pscores_data_lasso = f.summary_stats(pscores_lasso)
    print(pscores_data_lasso, '\n')
    
else: 
    print("Cross-validated penalty was strong enough that the best model is the intercept only")

# Export clean data
data['pscores'] = pscores.iloc[:, 0].values
CLEAN_DATA_NAME = 'clean_data.csv'
data.to_csv(PROCESSED_DATA_DIR / CLEAN_DATA_NAME, index=False)

########################################
# ATE ESTIMATE WITH INVERSE PROBABILITY WEIGHTING
########################################
data = pd.read_csv(PROCESSED_DATA_DIR / CLEAN_DATA_NAME)

ate_ipw = f.ATE_IPW(data, 'RXASP', 'FDEAD', 'pscores')
print("ATE estimate with IPW:\n", ate_ipw, "\n")

########################################
# CONDITIONAL AVG TREATMENT EFFECT ESTIMATE VIA OLS
########################################
data['SEXxRXASP'] = data['SEX']*data['RXASP'] # create the interaction variable

# Estimate CATE with OLS regression
OLS_cate_estimates = f.ols(OLS_covs, 'FDEAD', data, intercept=True)
print("OLS CATE Regression Output:\n", OLS_cate_estimates, '\n')

# Compute the std error for CATE men
OLS_vcov = f.ols(OLS_covs, 'FDEAD', data, intercept=True, vcov_out=True)[1] # variance-covariance matrix
cov_beta1_beta3 = OLS_vcov[2, 4] # Cov(beta1, beta3)
se_beta1_sq = np.diagonal(OLS_vcov)[2] # St error of beta1
se_beta3_sq = np.diagonal(OLS_vcov)[4] # St error of beta3
se_CATE_men = np.sqrt(se_beta1_sq + se_beta3_sq + 2 * cov_beta1_beta3)
print("Std Error for men's CATE:", se_CATE_men, "\n")

########################################
# INDIVIDUALIZED AVG TTEATMENT EFFECT ESTIMATE
########################################
data = data.drop(columns=['pscores']) # remove propensity scores
random.seed(RANDOM_SEED) # set seed

# initialize variables
var_y = 'FDEAD'
var_d = "RXASP"
var_x_ord  = ["AGE", "RSBP", "RDELAY", "SEX", "RCT", "RVISINF", "RATRIAL"]

# initialize a Modified Causal Random Forest
mcf = ModifiedCausalForest(
    var_y_name=var_y,
    var_d_name=var_d,
    var_x_name_ord=var_x_ord,
    gen_outpath=PROJECT_ROOT/"mcf_out",
    p_iate=True,
    p_iate_se=False
    )

# train the forest on the data
mcf.train(data)

# predict in-sample causal effects
results, _ = mcf.predict(data)

# extract the IATE df
iate_df = results["iate_data_df"]
print(iate_df.head())

# Plot histogram of IATE
plt.figure(figsize=(8, 5))
plt.hist(iate_df["fdead_lc1vs0_iate"], bins=30, edgecolor='black', alpha=0.3)
plt.xlabel("IATE")
plt.ylabel("Frequency")
plt.title("Histogram of IATE estimates")
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'IATE_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# descriptive analysis of IATE
diagnostics, _ = mcf.analyse(results)

# compute % patients with negative IATE
negative_share = (iate_df["fdead_lc1vs0_iate"] < 0).sum()/iate_df.shape[0]
print(f"Share negative: {negative_share:.2%}")

########################################
# HETEROGENEOUS EFFECTS ANALYSIS
########################################

# add the other covariates to iate_df
oth_cols = ['rasp3', 'rdef1', 'rdef2', 'rdef3', 'rdef4', 'rdef5', 'rdef6', 'rdef7', 'rdef8',
            'rconsc', 'stype_lacs', 'stype_pacs', 'stype_pocs', 'stype_oth']
oth_vars = data.loc[:, oth_cols].copy() # subset the original data frame

# merge with iate_df
iate_df = (
    iate_df
    .merge(oth_vars, how='inner', left_on='id_mcf', right_index=True)
)

# boxplot of iate for every categorical variable
cat_vars = ['sex', 'rct', 'rvisinf', 'ratrial'] + oth_cols

nrows, ncols = 6, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24))
axes = axes.flatten()

for ax, var in zip(axes, cat_vars):
    sns.boxplot(
        ax=ax,
        x=iate_df[var].astype('category'),
        y=iate_df['fdead_lc1vs0_iate'],
        color='lightblue',
        order=sorted(iate_df[var].dropna().unique())
    )

    ax.set_title(var)
    ax.set_xlabel(var)
    ax.set_ylabel('IATE')

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / 'iate_boxplot_cat.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()
    
# scatterplot of IATE for every continuous variable
cont_vars = ['age', 'rsbp', 'rdelay']

# 1x3 grid and colors
fig, axes = plt.subplots(
    1, len(cont_vars),
    figsize=(len(cont_vars) * 5, 4),
    sharey=True
)
colors   = ['red', 'green', 'blue']

for ax, var, col in zip(axes, cont_vars, colors):
    sns.regplot(
        x=var,
        y='fdead_lc1vs0_iate',
        data=iate_df,
        ax=ax,
        scatter_kws={'color': col, 'alpha': 0.05},
        line_kws={'color': 'darkgray'}
    )
    ax.set_title(f"IATE vs {var}")
    ax.set_xlabel(var)
    ax.set_ylabel('IATE')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'cont_vars_scatters.png', dpi=300, bbox_inches='tight')
plt.close()

# run OLS regressions on the trends
OLS_trends = {} # prepare empty dictionary for outputs

for var in cont_vars:
    OLS_trends[var] = f.ols([var], 'fdead_lc1vs0_iate', iate_df, intercept=True)
    print(f"Regression results for {var}:")
    print(OLS_trends[var])
    print("\n" + "-"*50 + "\n")


# EXTREME VALUES: inspecting the biggest beneficiaries and damaged
# compute the 5th and 95th percentile of IATE
lower = iate_df['fdead_lc1vs0_iate'].quantile(0.05)
upper = iate_df['fdead_lc1vs0_iate'].quantile(0.95)

# create top and bottom‐5% dummies
iate_df['bottom5pct'] = (iate_df['fdead_lc1vs0_iate'] <= lower).astype(int)
iate_df['top5pct']    = (iate_df['fdead_lc1vs0_iate'] >= upper).astype(int)


# DUMMIES' ANALYSIS
dummies = cat_vars.copy()
dummies.remove('rconsc') # rconsc is the only non-dummy categorical variable

# compute the means of the dummies for top and bottom
means_bottom = iate_df.loc[iate_df['bottom5pct']==1, dummies].sum()/iate_df.loc[iate_df['bottom5pct']==1,:].shape[0]
means_top    = iate_df.loc[iate_df['top5pct']==1,    dummies].sum()/iate_df.loc[iate_df['top5pct']==1,:].shape[0]

# create dataframe for comparison
dummies_comp = (
    pd.DataFrame({
        'dummy': dummies,
        'Bottom 5% mean': means_bottom.values,
        'Top 5% mean'   : means_top.values
    })
    .melt(id_vars='dummy',
          var_name='Group',
          value_name='Mean')
)

# barplot the means 
plt.figure(figsize=(14, 6))
sns.barplot(
    data=dummies_comp,
    x='dummy',
    y='Mean',
    hue='Group',
    palette= ['lightskyblue','mediumseagreen']
)
plt.title('Means of dummies in bottom vs top 5% of IATE')
plt.xlabel('Dummy Variable')
plt.xticks(rotation=-45)
plt.ylabel('Mean Value')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'means_dummies_top_vs_bottom.png', dpi=300, bbox_inches='tight')
plt.close()

# create comparison table
dummies_comp_table = (
    dummies_comp
    .pivot(index='dummy', columns='Group', values='Mean')
    .rename(columns={
    'Bottom 5% mean': 'bottom_5_mean',
    'Top 5% mean': 'top_5_mean'})
    .reset_index()
)

# add column for delta and sort
dummies_comp_table['delta'] = dummies_comp_table['top_5_mean'] - dummies_comp_table['bottom_5_mean']
dummies_comp_table = dummies_comp_table.sort_values(by='delta', ascending=False)
print("Means of dummy variables in top vs bottom 5\% of the IATE distribution:\n", dummies_comp_table, "\n")

# RCONSC ANALYSIS
# compute shares of each consciuosness state in upper and lower tails
bottom_props = (
    iate_df.loc[iate_df['bottom5pct'] == 1, 'rconsc']
    .value_counts(normalize=True)
    .sort_index())
top_props = (
    iate_df.loc[iate_df['top5pct'] == 1, 'rconsc']
    .value_counts(normalize=True)
    .sort_index())

# create data frame for barplot
rconsc_comp = (
    pd.DataFrame({
        'rconsc': bottom_props.index.astype(str),
        'Bottom 5%': bottom_props.values,
        'Top 5%':    top_props.values
    })
    .melt(id_vars='rconsc',
          var_name='Group',
          value_name='Share')
)

# substitute levels with their description
label_map = {'0': 'unconscious', '1': 'drowsy','2': 'fully alert'}
rconsc_comp['rconsc'] = rconsc_comp['rconsc'].map(label_map)

# barplot of consciousness states in top and bottom tails
plt.figure(figsize=(8, 6))
sns.barplot(
    data=rconsc_comp,
    x='rconsc',
    y='Share',
    hue='Group',
    palette=['lightblue', 'salmon'])

plt.title('Prevalence of consciousness states in top vs bottom 5% of IATE')
plt.xlabel('Consciousness State')
plt.ylabel('Share of observations')
plt.ylim(0, 1)
plt.legend(title='')
plt.tight_layout()

plt.savefig(FIGURES_DIR / 'rconsc_prevalence_top_vs_bottom.png', dpi=300, bbox_inches='tight')
plt.close()

# CONTINUOUS VARSIABLES' ANALYSIS
# plot histogram of continuous variables for top and bottom 5% of IATE
for var in cont_vars:
    bottom = iate_df[iate_df['bottom5pct'] == 1][var]
    top    = iate_df[iate_df['top5pct'] == 1][var]

    plt.figure()
    plt.hist(bottom, bins=30, alpha=0.4, label='Bottom 5% of IATE')
    plt.hist(top, bins=30, alpha=0.4, label='Top 5% of IATE')
    plt.legend()
    plt.title(f'{var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.savefig(FIGURES_DIR / f'iate_histogram_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()