"""
IST Causal Econometrics Project
Custom Functions
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from src.config import RANDOM_SEED


def nacount(data):
    """
    The function counts the number of missing values in each column of a dataframe.
    
    Parameters
    --------
    data: TYPE : pd.DataFrame
    DESCRIPTION: input dataframe for which we need to count the nas
    
    Returns
    --------
    missing: (TYPE: pd.DataFrame)
    DESCRIPTION: dataframe with the count and share of nas per each column of data
    """
    # extract the columns
    columns = data.columns
    
    # prepare empty dataframe
    missing = pd.DataFrame([], index = ['count', 'share'], columns = columns)
    
    # count nas per column
    for column in range(len(columns)):
        missing.iloc[0, column] = sum(data.iloc[:, column].isna())
        missing.iloc[1, column] = round(missing.iloc[0, column]/data.shape[0], 2)
        
    missing = missing.transpose()
    return missing

def mean_std(data, treatment):
    '''
    Computes mean and std of a the dataframe columns by treatment status

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataset
    treatment : TYPE: string
        DESCRIPTION: name of the treatment dummy
    
    Returns
    -------
    Returns a data frame with means and stds
    '''
    # subset data by treatment status
    treat = data.loc[data[treatment]==1, :] 
    control = data.loc[data[treatment]==0, :]
    
    # prepare empty df
    stds = pd.DataFrame([], columns = data.columns, index = ['Treatment mean','Control mean',
                                                             'Treatment std','Control std'])
    
    for var in data.columns:
        # compute means and std
        mt = treat[var].sum()/len(treat[var])
        mc = control[var].sum()/len(control[var])
        sdt = np.sqrt(sum((xi - mt) ** 2 for xi in treat[var]) / len(treat[var]))
        sdc = np.sqrt(sum((xi - mc) ** 2 for xi in control[var]) / len(control[var]))
        
        # store values
        stds.loc['Treatment mean', var] = mt
        stds.loc['Control mean', var] = mc
        stds.loc['Treatment std', var] = sdt
        stds.loc['Control std', var] = sdc
    
    stds = stds.transpose()
    
    return stds

def meandiff(data, treatment, var):
    """
    Compares the mean betweenn treatment and control of var

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataset
    treatment : TYPE: string
        DESCRIPTION: name of the treatment dummy
    variables : TYPE: string
        DESCRIPTION: name of the variable for the mean comparison

    Returns
    -------
    Returns a data frame with:
            - treatment, control means, 
            - difference, 
            - standard error, 
            - t-value, p-value
    """
    treat = data.loc[data[treatment]==1, var] 
    control = data.loc[data[treatment]==0, var]
    meantreatment = treat.mean()
    meancontrol = control.mean() 
    meandiff = meantreatment - meancontrol
    std = (np.sqrt(treat.var() / len(treat) + control.var() / len(control))) 
    tval = meandiff / std
    pval = stats.norm.sf(abs(tval)) * 2
    
    result = pd.DataFrame({'Treatment' : meantreatment, 
                           'Control' : meancontrol, 
                           'Mean diff' : meandiff,
                           'Std' : std,
                           't-value' : tval,
                           'p-value' : pval}, index = ['Comparison'])
    return result
    
def ols(covariates, response, data, intercept=True, vcov_out=False):
    '''
    The function estimates the betas of a linear model with OLS and computes
    some key statistics
    
    Parameters
    ----------
    covariates: list of covariates to include in the model
    response: string with the response variable
    data : pd.DataFrame with the dataset
    intercept : boolean for including or not the intercept
    vcov_out : boolean for including or not the variance-covariance matrix in the output

    Returns
    -------
    Tuple with pd.DataFrame with the coefficient estimate, the standard error, the t-statistics
    and p-value (2-sided test against 0) for each covariate and array with vcov matrix
    '''
    # extract the data and convert to numpy
    X = data[covariates].values
    y = data[response].values
    
    if intercept:
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1) # include constant col

    # Estimate beta
    betahat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    # Extract key quantities
    n = X.shape[0] # n. observations
    p = X.shape[1] # n. covariates
    fitted = X.dot(betahat) # fitted values
    residuals = y - fitted # residuals
    s2 = (residuals.T.dot(residuals))/(n-p) # variance of error estimate 
    vcov = s2*inv(X.T.dot(X)) # variance-covariance matrix
    sehat = np.sqrt(np.diagonal(vcov)) # standard errors
    
    # preprare output dataframe
    if intercept:
        out = pd.DataFrame([], index = ['intercept'] + covariates, 
                           columns=['coeff', 'se', 't-value', 'p-value'])

    else:
        out = pd.DataFrame([], index = X.columns, 
                           columns=['coeff', 'se', 't-value', 'p-value'])
    
    out['coeff'] = betahat
    out['se'] = sehat
    
    # compute t-values and p-values
    for var in range(p):
        t_value = betahat[var]/sehat[var]
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), n-p))
        
        # store in out
        out.iloc[var, 2] = t_value
        out.iloc[var, 3] = p_value
    if vcov_out:
        return out, vcov
    else:
        return out

def lasso_selection(data, dependent_var):
    """
    Estimate a Lasso model using cross-validation to select the best penalty parameter,
    and return the names of covariates with non-zero coefficients as a string

    Parameters:
    - data: pd.DataFrame with the dataset.
    - dependent_var: the name of the dependent variable column in data.

    Returns:
    - A string containing the selected covariates
    """
    # prepare data
    X = data.drop(columns=[dependent_var])
    y = data[dependent_var]
    
    # Fit Lasso model with 5-fold CV
    model = LassoCV(cv=5, random_state=0).fit(X, y)
    
    # Get the coefficients from the model
    coef = model.coef_
    
    # identify features with non-zero coefficients
    selected_features = X.columns[coef != 0].tolist()
    
    # convert the list of selected features into a string
    selected_features_str = ', '.join(selected_features)
    
    return selected_features_str

def logit_pscores(treatment, covariates, data, intercept=True):
    '''
    The function uses a Logistic regression to estimate the propensity scores
    
    Parameters
    ----------
    treatment: string with the treatment dummy
    covariates: list of covariates to include in the Logit model
    data : pd.DataFrame with the dataset
    intercept : boolean for including or not the intercept

    Returns
    -------
    series with the predicted propensity scores for each individual in data
    '''
    y = data[treatment].values
    
    if intercept:
        X = sm.add_constant(data[covariates]) # add constant
    else:
        X = data[covariates]
    
    X = X.values
    
    # run the Logit and predict the pscores on X
    pscores = sm.Logit(endog=y, exog=X).fit(disp=0).predict()
    
    return pscores

def ATE_IPW(data, treatment, outcome, pscores, 
            n_bootstrap = 1000, random_state = RANDOM_SEED):
    """
    Computes ATE by inverse probability weighting and its standard
    error with bootstrap

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data on which balancing checks should be conducted
    treatment : TYPE: string
        DESCRIPTION: name of the binary treatment variable
    outcome : TYPE: string
        DESCRIPTION: name of the outcome variable
    pscores: TYPE: string
        DESCRIPTION: name of the columns of estimates propensity scores
    n_bootstrap : TYPE: int, optional
        Number of bootstrap samples to draw for estimating the standard error
    random_state : TYPE: int, optional
        Seed for the random number generator

    Returns
    -------
    pd.DataFrame the ATE estimated with IPW, its std error, t-val and p-val
    """
    # extract key quantities
    N = data.shape[0]
    d = data[treatment]
    y = data[outcome]
    p = data[pscores]
    
    # compute ate
    treated_term = (d * y / p).sum()
    control_term = ((1 - d) * y / (1 - p)).sum()

    ate_ipw = (treated_term - control_term) / N
    
    # bootstrap for standard error
    rng = np.random.RandomState(random_state)
    boot_ates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        bs_idx = rng.choice(N, size=N, replace=True)
        bs = data.iloc[bs_idx]
        d_b = bs[treatment]
        y_b = bs[outcome]
        p_b = bs[pscores]
        treated_b = (d_b * y_b / p_b).sum()
        control_b = ((1 - d_b) * y_b / (1 - p_b)).sum()
        boot_ates[i] = (treated_b - control_b) / N

    # compute the standard error
    mean_bs = boot_ates.sum() / n_bootstrap
    squared_diffs = (boot_ates - mean_bs) ** 2
    variance_bs = squared_diffs.sum() / (n_bootstrap - 1)
    se = np.sqrt(variance_bs)
    
    # compute t-value and p-value
    t_val = ate_ipw / se
    p_val = stats.norm.sf(abs(t_val)) * 2
    
    # store in df
    ate_ipw_df = pd.DataFrame({'Point estimate' : ate_ipw, 
                               'STD Error' : se,
                               't-value' : t_val, 
                               'p-value' : p_val}, index = ['ATE IPW'])
    
    return ate_ipw_df
    
def summary_stats(data):
    """
    Summary stats: mean, variance, standard deviation, maximum and minimum.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe for which descriptives will be computed
    Returns
    -------
    my_descriptives: TYPE: pd.DataFrame
    DESCRIPTION: dataframe with descriptive statistics per each column of data
    """
    # generate storage for the stats as an empty dictionary
    my_descriptives = {}
    # loop over columns
    for col_id in data.columns:
        # fill in the dictionary with descriptive values by assigning the
        # column ids as keys for the dictionary
        my_descriptives[col_id] = [data[col_id].mean(),                  # mean
                                   data[col_id].var(),               # variance
                                   data[col_id].std(),                # st.dev.
                                   data[col_id].max(),                # maximum
                                   data[col_id].min(),                # minimum
                                   sum(data[col_id].isna()),          # missing
                                   len(data[col_id].unique()),  # unique values
                                   data[col_id].shape[0]]      # number of obs.
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=['mean', 'var', 'std', 'max', 'min',
                                          'na', 'unique', 'obs']).transpose()
    # define na, unique and obs as integers such that no decimals get printed
    ints = ['na', 'unique', 'obs']
    # use the .astype() method of pandas dataframes to change the type
    my_descriptives[ints] = my_descriptives[ints].astype(int)
    
    return my_descriptives

def balance_check(data, treatment, variables):
    """
    Check covariate balance.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data on which balancing checks should be conducted
    treatment : TYPE: string
        DESCRIPTION: name of the binary treatment variable
    variables : TYPE: tuple
        DESCRIPTION: names of the variables for balancing checks

    Returns
    -------
    Returns and Prints the Table of Descriptive Balancing Checks
    """
    # create storage for output as an empty dictionary for easy value fill
    balance = {}
    # loop over variables
    for varname in variables:
        # define according to treatment status by logical vector of True/False
        # set treated and control apart using the location for subsetting
        # using the .loc both labels as well as booleans are allowed
        treated = data.loc[data[treatment] == 1, varname]
        control = data.loc[data[treatment] == 0, varname]
        # compute difference in means between treated and control
        mdiff = treated.mean() - control.mean()
        # compute the corresponding standard deviation of the difference
        # squared root of sum of variance of treated scaled by the number of
        # treated + variance of control scaled by the number of controls
        mdiff_std = (np.sqrt(treated.var() / len(treated)
                     + control.var() / len(control)))
        # compute the t-value for the difference
        mdiff_tval = mdiff / mdiff_std
        # compute pvalues based on the normal distribution (requires scipy)
        # sf stands for the survival function (also defined as 1 - cdf)
        mdiff_pval = stats.norm.sf(abs(mdiff_tval)) * 2  # twosided
        # compute the standardized difference
        sdiff = abs(mdiff / np.sqrt((treated.var() + control.var()) / 2)) * 100
        # combine values
        balance[varname] = [treated.mean(), control.mean(),
                            mdiff, mdiff_std, mdiff_tval, mdiff_pval, sdiff]
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    balance = pd.DataFrame(balance,
                           index=["Treated", "Control", "MeanDiff", "Std",
                                  "tVal", "pVal", "StdDiff"]).transpose()
    # print the descriptives (\n inserts a line break)
    #print('Balancing Checks:', '-' * 80,
    #      round(balance, 2), '-' * 80, '\n\n', sep='\n')
    # return results
    return balance 