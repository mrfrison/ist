# IST

> The following in an econometric analysis of the IST data (one of the
> largest clinical trials ever conducted) to investigate the effect of
> aspirin on stroke outcomes and explore treatment heterogeneity,
> particularly focusing on survival and disability at six months from
> treatment assignment.

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python
3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

-   [Overview](#overview)
-   [Repository Structure](#structure)
-   [Requirements](#requirements)
-   [Installation](#installation)
-   [Data](#data)
-   [Data Cleaning](#cleaning)
-   [Methodology](#methodology)
-   [Results](#results)

## Overview

Stroke is a leading cause of death and disability worldwide, making the
search for effective treatments a critical area of medical research. The
International Stroke Trial (IST) was one of the largest clinical trials
ever conducted in acute stroke, enrolling 19,435 patients from 466
centers across the globe. This randomized controlled trial (RCT) aimed
to assess the effects of aspirin and heparin—alone or in com-
bination—on stroke outcomes, particularly focusing on survival and
disability at six months.

## Repository Structure

```         
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python package dependencies
├── .gitignore                   # Files to ignore in version control
│
├── data/                        # Data directory
│   ├── raw/                    # Original data
│   └── processed/              # Cleaned, analysis-ready data
│
├── src/                         # Source code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration and paths
│   └── functions.py            # Custom functions
│
├── scripts/                     # Analysis scripts
│   ├── main_analysis.py        # Main analysis
│
└── output/                      # Generated outputs
   ├── figures/                # Plots
   └── tables/                 # Result tables
```

## Requirements

-   Python 3.8 or higher
-   See `requirements.txt` for package dependencies

**Key packages:** - `numpy` - Numerical computations - `pandas` - Data
visualisations - `matplotlib` - Statistical models - `statsmodels` -
Modified Causal Forest - `ModifiedCausalForest`

## Installation

**Clone the repository**

``` bash
git clone https://github.com/mrfrison/ist.git
cd ist
```

## Data

**Primary Dataset**: The International Stroke Trial database -
<https://pmc.ncbi.nlm.nih.gov/articles/PMC3104487/>

The dataset analysed consists of the following variables:

| Variable | Explanation |
|-------------------------------|----------------------------------------|
| RXASP | **Treatment:** Trial aspirin allocated (Y/N) |
| FDEAD | **Outcome:** Dead at six-month follow-up (Y/N) |
| AGE | Age in years |
| RSBP | Systolic blood pressure at randomisation (mmHg) |
| RDELAY | Delay between stroke and randomisation in hours |
| SEX | M = male; F = female |
| RCT | CT before randomisation (Y/N) |
| RVISINF | Infarct visible on CT (Y/N) |
| RATRIAL | Atrial fibrillation (Y/N) |
| RASP3 | Aspirin within 3 days prior to randomisation (Y/N) |
| DASP14 | Aspirin given for 14 days or till death or discharge (Y/N) |
| RDEF1 | Face deficit (Y/N/C = can't assess) |
| RDEF2 | Arm/hand deficit (Y/N/C = can't assess) |
| RDEF3 | Leg/foot deficit (Y/N/C = can't assess) |
| RDEF4 | Speech deficit (Y/N/C = can't assess) |
| RDEF5 | Visual deficit (Y/N/C = can't assess) |
| RDEF6 | Visuospatial disorder (Y/N/C = can't assess) |
| RDEF7 | Neurological deficits (Y/N/C = can't assess) |
| RDEF8 | Other deficit (Y/N/C = can't assess) |
| RCONSC | Conscious state at randomisation (F = fully alert, D = drowsy, U = unconscious) |
| STYPE | Stroke subtype (TACS/PACS/POCS/LACS/other) |
| COUNTRY | Abbreviated country code |


## Data Cleaning

-   **Missing values**: I coded a function that inspects the dataset for
    NAs. It showed that, for the variables RATRIAL and RASP3, there are
    1033 NAs (5% of the observations) and 21 for DASP14 ( \< 1%). Given
    the very small shares of missing values, I decided to drop these
    observations.

-   **Categorical variables**: I recoded the categorical variables with
    Y/N levels as 1/0 dummies, and SEX as 1 if M, 0 if F. I encoded the
    levels of RCONSC into an increasing numerical sequence (’F’ = 2, ’D’
    = 1, ’U’ = 0). I converted the COUNTRY and STYPE variables into
    dummies. While examinimg the unique values of DASP14, I noticed the
    presence of mislabeled values (’U’ instead of Y/N). I removed the
    mislabeled observations. Finally, to recode the variables with Y/N/C
    levels, I decided to substitute C with the outcomes of a
    *B**e**r*(*p*), whith *p* chosen based on the empirical probability
    of a "yes". Having no way to make sure if "can’t assess" is half way
    trough "yes" and "no", or closer to one of the two, I believe the
    best approach is to mimic the distribution of the clean data. This
    comes at the expense of potentially inflating the standard errors of
    the estimators, but minimizes the bias and preserves all the other
    data points of the patients, that are otherwise clean.

-   **Outliers**: I plotted the boxplots of the three continuous
    variables (AGE, RSBP, RDELAY) in Figure
    <a href="#fig:boxplots" data-reference-type="ref"
    data-reference="fig:boxplots">1</a>. From the age boxplot we can see
    that some patients were registers as 130 years old or older. These
    are for sure errors, so I removed said observations. The boxplot of
    RSBP reveals 60 patients with systolic pressure greater than 250
    mmHg, which is extremely high, but still possible. RDELAY is well
    behaved.

<img src="https://github.com/user-attachments/assets/34cdd24e-db06-4639-a310-217da323a306" alt="image" width="723.5" height="589"/>

Table <a href="#tab:means_std_treat_control" data-reference-type="ref"
data-reference="tab:means_std_treat_control">1</a> reports the means and
standard deviations on all variables in the final clean dataset per
treatment status.

| Variable | Treat Mean | Treat Std | Control Mean | Control Std | Variable | Treat Mean | Treat Std | Control Mean | Control Std |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| RXASP | 1.00 | 0.00 | 0.00 | 0.00 | ctry_FINL | 0.00 | 0.05 | 0.00 | 0.05 |
| FDEAD | 0.22 | 0.42 | 0.23 | 0.42 | ctry_FRAN | 0.00 | 0.01 | 0.00 | 0.01 |
| AGE | 77.26 | 25.87 | 76.61 | 24.61 | ctry_GREE | 0.01 | 0.09 | 0.01 | 0.09 |
| RSBP | 160.06 | 27.75 | 160.46 | 27.67 | ctry_HONG | 0.01 | 0.08 | 0.01 | 0.07 |
| RDELAY | 20.04 | 12.40 | 20.04 | 12.52 | ctry_HUNG | 0.01 | 0.07 | 0.01 | 0.08 |
| SEX | 0.53 | 0.50 | 0.54 | 0.50 | ctry_INDI | 0.01 | 0.11 | 0.01 | 0.11 |
| RCT | 0.67 | 0.47 | 0.67 | 0.47 | ctry_ISRA | 0.01 | 0.08 | 0.01 | 0.07 |
| RVISINF | 0.32 | 0.47 | 0.33 | 0.47 | ctry_ITAL | 0.17 | 0.38 | 0.17 | 0.38 |
| RATRIAL | 0.17 | 0.38 | 0.17 | 0.37 | ctry_JAPA | 0.00 | 0.02 | 0.00 | 0.02 |
| RASP3 | 0.21 | 0.41 | 0.22 | 0.41 | ctry_NETH | 0.04 | 0.19 | 0.04 | 0.20 |
| DASP14 | 0.92 | 0.27 | 0.02 | 0.12 | ctry_NEW | 0.02 | 0.15 | 0.02 | 0.15 |
| RDEF1 | 0.73 | 0.44 | 0.74 | 0.44 | ctry_NORW | 0.03 | 0.17 | 0.03 | 0.17 |
| RDEF2 | 0.86 | 0.34 | 0.86 | 0.34 | ctry_POLA | 0.04 | 0.20 | 0.04 | 0.20 |
| RDEF3 | 0.77 | 0.42 | 0.77 | 0.42 | ctry_PORT | 0.02 | 0.14 | 0.02 | 0.14 |
| RDEF4 | 0.45 | 0.50 | 0.46 | 0.50 | ctry_ROMA | 0.00 | 0.03 | 0.00 | 0.03 |
| RDEF5 | 0.19 | 0.40 | 0.20 | 0.40 | ctry_SING | 0.01 | 0.08 | 0.01 | 0.09 |
| RDEF6 | 0.20 | 0.40 | 0.20 | 0.40 | ctry_SLOK | 0.00 | 0.07 | 0.00 | 0.07 |
| RDEF7 | 0.12 | 0.33 | 0.12 | 0.32 | ctry_SLOV | 0.00 | 0.05 | 0.00 | 0.05 |
| RDEF8 | 0.07 | 0.25 | 0.07 | 0.25 | ctry_SOUT | 0.00 | 0.06 | 0.00 | 0.06 |
| RCONSC | 1.75 | 0.46 | 1.75 | 0.46 | ctry_SPAI | 0.03 | 0.16 | 0.03 | 0.16 |
| ctry_ARGE | 0.03 | 0.16 | 0.03 | 0.17 | ctry_SRI | 0.00 | 0.03 | 0.00 | 0.03 |
| ctry_AUSL | 0.03 | 0.17 | 0.03 | 0.17 | ctry_SWED | 0.03 | 0.18 | 0.03 | 0.18 |
| ctry_AUST | 0.01 | 0.11 | 0.01 | 0.11 | ctry_SWIT | 0.09 | 0.29 | 0.09 | 0.29 |
| ctry_BELG | 0.01 | 0.12 | 0.01 | 0.12 | ctry_TURK | 0.02 | 0.12 | 0.01 | 0.12 |
| ctry_BRAS | 0.00 | 0.07 | 0.00 | 0.06 | ctry_UK | 0.32 | 0.46 | 0.31 | 0.46 |
| ctry_CANA | 0.01 | 0.08 | 0.01 | 0.08 | STYPE_LACS | 0.24 | 0.43 | 0.24 | 0.43 |
| ctry_CHIL | 0.00 | 0.06 | 0.00 | 0.05 | STYPE_OTH | 0.00 | 0.05 | 0.00 | 0.05 |
| ctry_CZEC | 0.02 | 0.15 | 0.02 | 0.15 | STYPE_PACS | 0.40 | 0.49 | 0.41 | 0.49 |
| ctry_DENM | 0.00 | 0.04 | 0.00 | 0.04 | STYPE_POCS | 0.12 | 0.32 | 0.11 | 0.32 |
| ctry_EIRE | 0.00 | 0.05 | 0.00 | 0.05 | STYPE_TACS | 0.24 | 0.43 | 0.24 | 0.43 |

Table 1: Means and standard deviations by treatment status

## Methodology

### ATE estimate by means difference

Table <a href="#tab:meandiff" data-reference-type="ref"
data-reference="tab:meandiff">2</a> reports the result of the comparison
of the means of FDEAD between treatment and control group. It shows that
while in the treatment group 22.1% of the patients died within six
months, in the control group 23.3% did. This suggests that the treatment
decreases the probability of death within six months by 1.2%. The
standard error, t-statistic, and p-value of the mean difference show
that the two means are statistically different at the 5% level. The
identifying assumptions in this framework are:

-   Stable Unit Treatment Value Assumption (SUTVA):
    *Y*<sub>*i*</sub> = *D*<sub>*i*</sub>*Y*<sub>*i*</sub>(1) + (1 − *D*<sub>*i*</sub>)*Y*<sub>*i*</sub>(0),
    and

-   Complete randomization:
    (*Y*<sub>*i*</sub>(1), *Y*<sub>*i*</sub>(0))⊥*D*<sub>*i*</sub> for
    *i* = 1, ..., *n*.

The first assumption is valid, as there is no way in which treating a
patient with aspirin can affect the treatment assignment of another
patient, nor the other’s death within six months. The second assumption
is more problematic. The only significant difference between treatment
and control that we can see from Table
<a href="#tab:means_std_treat_control" data-reference-type="ref"
data-reference="tab:means_std_treat_control">1</a> is for the variable
DASP14. In the treatment group, 92% of the patients received aspirin for
14 days or till death or discharge, while in the control group only 2%
did. Testing the difference between the two means we obtain a standard
error of 0.00306 and a t-statistic of 297.28. This is strong evidence
that the two groups differ under DASP14. However, DASP14 is a
post-treatment variable that apparently depends on treatment assignment,
but that does not influence it. In light of this, it seems that the
complete randomization assumption also holds.

|            | Treatment |  Control | Mean diff |      Std |   t-value |  p-value |
|:-----------|----------:|---------:|----------:|---------:|----------:|---------:|
| Comparison |  0.221472 | 0.233379 | -0.011906 | 0.006048 | -1.968567 | 0.049003 |

Table 2: Mean difference in FDEAD

### ATE estimate with OLS regression

Table <a href="#tab:OLS_ATE" data-reference-type="ref"
data-reference="tab:OLS_ATE">3</a> reports the results of the OLS
regression. From it see that the ATE estimate became positive and
statistically significant, suggesting that being allocated to the trial
with aspirin raises mortality by 7.53%.

This apparent contradiction with the result obtained before is due to
the inclusion in the OLS model of the other covariates, especially
DASP14. The latter reports whether the patient actually took aspirin or
not, no matter if assigned to the treatment group. In this OLS model,
the coefficient of DASP14 (-0.096143, se 0.014286) is what represents
the aspirin effect. The coefficient of RXASP can be interpreted as the
additional effect of being allocated to the trial, given that the
individual took aspirin for 14 days or until death or discharge.
Ultimately, the reason why RXASP has a positive coefficient is that
there is a 7.62% chance of not being treated if allocated to treatment.
In fact, we see from Table
<a href="#tab:means_std_treat_control" data-reference-type="ref"
data-reference="tab:means_std_treat_control">1</a> that in the treatment
group 92.38% of the patients were actually treated with aspirin, while
7.62% were not. The combined effect of being allocated to trial *and*
taking aspirin for 14 days or until death or discharge is a -2.1%
reduction on the probability of death, much closer to the ATE estimated
by difference in means.

|           | Coefficient  |  Std error   |   t-Value    |   p-Value    |
|:----------|:------------:|:------------:|:------------:|:------------:|
| intercept |  -0.268207   |   0.026834   |  -9.994856   |   0.000000   |
| **RXASP** | **0.075260** | **0.014253** | **5.280413** | **0.000000** |
| DASP14    |  -0.096143   |   0.014286   |  -6.730100   |   0.000000   |
| AGE       |   0.008669   |   0.000270   |  32.153226   |   0.000000   |
| RSBP      |  -0.000727   |   0.000107   |  -6.798539   |   0.000000   |
| RDELAY    |  -0.002297   |   0.000243   |  -9.469209   |   0.000000   |
| SEX       |  -0.005681   |   0.006029   |  -0.942236   |   0.346084   |
| RCT       |  -0.006504   |   0.007292   |  -0.891852   |   0.372484   |
| RVISINF   |   0.078983   |   0.007283   |  10.845204   |   0.000000   |
| RATRIAL   |   0.139036   |   0.008049   |  17.274453   |   0.000000   |

Table 3: OLS estimate of ATE

### Estimates of the propensity scores

I estimate the propensity score using a logistic regression model with
observed covariates. To choose the covariates on which to run the
logistic regression that estimates the propensity scores, I used two
methods: (1) I used all the covariates used for the OLS ATE estimate
except DASP14 (which is not a pre-treatment observation, but rather an
ex-post observation on the treatment received), and (2) I used a Lasso
model to select the best predictors of treatment assignment. Both
methods yield propensity scores very close to 0.5 (see Table
<a href="#tab:pscores_stats" data-reference-type="ref"
data-reference="tab:pscores_stats">4</a>), which is the share of
patients assigned to the treatment group. Thus, the results do not
deviate from my previous answer. In light of this, we can conclude that
the randomization was effective.

### ATE estimate by inverse probability weighting

I used inverse probability weighting (IPW) to estimate the ATE and
bootstrapping to compute its standard error. In a RCT, like the IST, we
can use mean difference to estimate the ATE, because unconfoundness is
achieved by design, through randomization. The IPW estimator of the ATE
achieves unconfoundness, even when treatment is not randomly assigned,
by weighting differently patients’ outcomes based on their probability
to be assignment to the treatment group. If we have a good estimate of
the propensity scores, the IPW estimator produces an unbiased estimate
of the ATE. With respect to OLS with covariates, if the linearity
assumption is violated, IPW leads to a lower bias in the ATE estimate.
In this case, given that the propensity scores range from 0.484 to
0.510, there are also no instability issues due to extreme propensity
scores.

**Logit with AGE, RSBP, RDELAY, SEX, RCT, RVISINF, RATRIAL** \| Variable
\| mean \| var \| std \| max \| min \| range \| obs \| \| -------- \|
-------- \| -------- \| -------- \| -------- \| -------- \| ------ \|
----- \| \| pscores \| 0.499836 \| 0.000084 \| 0.009175 \| 0.531731 \|
0.471809 \| 0.0599 \| 18266 \|

**Logit with Lasso-selected covariates** \| Variable \| mean \| var \|
std \| max \| min \| range \| obs \| \| -------- \| -------- \| --------
\| -------- \| -------- \| -------- \| ------ \| ----- \| \| pscores \|
0.499836 \| 0.000011 \| 0.003243 \| 0.510387 \| 0.484080 \| 0.0263 \|
18266 \|

Table 4: Descriptive statistics of the propensity scores

Table <a href="#tab:ATE_ipw" data-reference-type="ref"
data-reference="tab:ATE_ipw">5</a> reports the ATE estimated with IPW
and its standard error. The point estimate is -0.012 showing that
aspirin slightly reduces the probability of death, in line with the IST
findings. Its standard error of 0.007 indicates that the ATE estimate is
significant at the 10% level. This results are in line with the IST
findings, although the significance level is moderate. The IPW method
produces the same point estimate I obtained with mean difference (as
expected, given the high homogeneity of the propensity scores). However
it departs from the OLS coefficient of RXASP (-0.075). Furthermore, the
significance level was higher with both of the previous methods. This
discrepancy might be due to the bootstrapping procedure used to compute
the standard errors.

|         | Point estimate | STD Error |   t-value |  p-value |
|:--------|---------------:|----------:|----------:|---------:|
| ATE IPW |      -0.012123 |  0.007029 | -1.724758 | 0.084571 |

Table 5: ATE estimate by inverse probability weighting

### CATE estimate via OLS for men and women

Because
CATE<sub>*m**e**n*</sub> = *E*$$*Y*<sup>1</sup> − *Y*<sup>0</sup>\|*S**E**X* = 1$$
and
CATE<sub>*w**o**m**e**n*</sub> = *E*$$*Y*<sup>1</sup> − *Y*<sup>0</sup>\|*S**E**X* = 0$$,
we can estimate CATE for men and women through the following linear
model:
FDEAD<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>RXASP<sub>*i*</sub> + *β*<sub>2</sub>SEX<sub>*i*</sub> + *β*<sub>3</sub>SEX<sub>*i*</sub> ⋅ RXASP<sub>*i*</sub> + \*γ\*\*X<sub>i</sub> + ε<sub>i\*</sub>.
In fact:

-   CATE for women: the model becomes
    FDEAD<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>RXASP<sub>*i*</sub> + *γ**X***<sub>i</sub> + ε<sub>i</sub>,
    so
    E$$*Y*\|*D* = 1, *S**E**X* = 0$$ − E$$*Y*\|*D* = 0, *S**E**X* = 0$$ = β<sub>0</sub> + β<sub>1</sub> + γX<sub>*i*</sub> − (*β*<sub>0</sub> + \*γ\*\*X<sub>i</sub>) = β\*<sub>1</sub>

-   CATE for men: the model becomes
    FDEAD<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>2</sub> + (*β*<sub>1</sub> + *β*<sub>3</sub>)RXASP<sub>*i*</sub> + *γ**X***<sub>i</sub> + ε<sub>i</sub>,
    so
    E$$*Y*\|*D* = 1, *S**E**X* = 1$$ − E$$*Y*\|*D* = 0, *S**E**X* = 1$$ = β<sub>0</sub> + β<sub>2</sub> + (β<sub>1</sub> + β<sub>3</sub>) + γX<sub>*i*</sub> − (*β*<sub>0</sub> + *β*<sub>2</sub> + \*γ\*\*X<sub>i</sub>) = β<sub>1</sub> + β\*<sub>3</sub>

Table <a href="#tab:CATE_OLS" data-reference-type="ref"
data-reference="tab:CATE_OLS">6</a> reports the results of the OLS
regression. The coefficient *β̂*<sub>1</sub> is the CATE estimate for
women and the sum of *β̂*<sub>1</sub> + *β̂*<sub>3</sub> is the CATE
estimate for men. $\widehat{CATE}\_{women} = \hat \beta_1 = -0.007917$
and has a standard error of 0.008619, thus it is not statistically
different from zero.
$\widehat{CATE}\_{men} = \hat \beta_1 + \hat \beta_3 = -0.015792$. The
standard error of men’s CATE can be computed as:

$$SE(\widehat{CATE}\_{men}) = \sqrt{SE(\hat \beta_1)^2 + SE(\hat \beta_3)^2 +2\text{Cov}(\hat \beta_1, \hat \beta_3)} = 0.00848.$$

This produces a t-statistic of -1.86 and a p-value of 0.0626, therefore
the CATE on the probability of death for men is statistically
significant at the 10% level.

To analyze gender heterogeneity we can compute the difference between
men and women’s CATEs:
$\widehat{CATE}\_{men} - \widehat{CATE}\_{women} = \hat \beta_1 + \hat \beta_3 - \hat \beta_1 = \hat \beta_3 = -0.0079$
and its standard error, which is 0.0118. Since the standard error is
greater than the point estimate, we can conclude that there is no
statistically significant difference between the two. From the small
value of the point estimate and the absence of significance we can
conclude that there is no heterogeneity regarding gender.

| Variable      | Coefficient   | Std error    | t-Value       | p-Value      |
|---------------|---------------|--------------|---------------|--------------|
| intercept     | -0.269360     | 0.027036     | -9.962961     | 0.000000     |
| **RXASP**     | **-0.007917** | **0.008619** | **-0.918460** | **0.358390** |
| **SEX**       | **-0.001675** | **0.008442** | **-0.198417** | **0.842721** |
| **SEXxRXASP** | **-0.007875** | **0.011801** | **-0.667324** | **0.504573** |
| AGE           | 0.008673      | 0.000270     | 32.129496     | 0.000000     |
| RSBP          | -0.000726     | 0.000107     | -6.787228     | 0.000000     |
| RDELAY        | -0.002330     | 0.000243     | -9.597218     | 0.000000     |
| RCT           | -0.010038     | 0.007282     | -1.378430     | 0.168087     |
| RVISINF       | 0.079625      | 0.007291     | 10.920726     | 0.000000     |
| RATRIAL       | 0.139115      | 0.008059     | 17.262427     | 0.000000     |

Table 6: CATE estimate with OLS

### IATE estimate

Figure <a href="#fig:IATE_hist" data-reference-type="ref"
data-reference="fig:IATE_hist">2</a> reports the histogram of the IATEs
estimated with a Modified Causal Forest. We can see that the
distribution of the IATEs is bell-shaped, with 49.97% of the estimates
being negative. The standard deviation is 0.065, and IATE values span
from a minimum of -0.246 to a maximum of 0.257. Therefore, the ATE
estimate masks some variability across patients. For instance, while
IATEs that lie below the 5th percentile of the distribution imply that
the patient benefits from a reduction of death probability in excess of
-0.106, those above the 95th percentile show an increase greater than
0.111.

Using a modified causal forest is useful when i) the confounders are
either too many for a classical parametric approach to be applicable (it
solves the high-dimensionality problem) or ii) their effects on the
outcome and treatment variables cannot be easily modeled by making
assumptions on a functional form. Thus, unlike mean difference, it can
be used outside the context of RCTs. Furthermore, with respect to OLS, a
Modified Causal Forest can handle a greater number of covariates and
nonlinearities of the effects. Moreover, by splitting trees on features
that generate treatment heterogeneity, a modified causal forest computes
CATE directly. This allows to uncover heterogeneous treatment effects
(something that we cannot do with mean difference) without resorting to
interaction variables like with OLS. In particular, with Modified Causal
Forests we can compute individualized average treatment effects (IATEs),
which are CATEs at the finest possible aggregation level. This is useful
to guide personalized policy, targeted marketing, or risk-stratified
medicine with granular data on how the treatement affects the various
subgroups of the population.

## Results

### Heterogeneity in effects
The most relevant estimates to understand treatment heterogeneity are
those that are far from the ATE. This is because i) they are more likely
to be significantly different from zero, and ii) they are related to
extreme treatment effects, so they can shed light on population
subgroups which either extensively benefit from treatment or are
seriously harmed.

To understand if the treatment has heterogeneous effects we should look
at the variables that might plausibly interact with aspirin’s effect. It
seems natural to exclude country dummies. Even if there was
heterogeneity in the effects of aspirin among countries, it would depend
on other features that are incidentally more prevalent among the
patients of that nationality. On the other hand, all physiological
characteristics of the patient (AGE, RSBP, SEX, RVISINF, RATRIAL, all
the deficits’ variables, RCONSC, STYPE) and variables detailing
hospitalization and diagnostic procedures (RASP3, RDELAY, RCT) are
relevant in relation to possible treatment heterogeneity.

To begin with I inspected the dummy and categorical variables. For each
one of them, I created boxplots to see how the distribution of the IATE
estimates changed across the levels. These boxplots are reported in
Figure <a href="#fig:IATE_cat" data-reference-type="ref"
data-reference="fig:IATE_cat">3</a>. We can see that the inter-quartile
ranges of IATE for all the dummy and categorical variables are
overlapping across the levels. Therefore, we can conclude that there are
only minor differences in the distributions and that heterogeneity is
moderate.

Next, I created scatter plots for the continuous variables (AGE, RSBP,
and RDELAY) against the estimates of the IATE (see Figure
<a href="#fig:cont_vars_scatters" data-reference-type="ref"
data-reference="fig:cont_vars_scatters">4</a>) to highlight any
correlation. We observe slight negative trends of IATE with respect to
the continuous variables, visualized by the gray fitted lines. Table
<a href="#tab:reg_IATE_cont_vars" data-reference-type="ref"
data-reference="tab:reg_IATE_cont_vars">7</a> reports the outcomes of
simple linear regressions of IATE against each continuous variable. In
all the cases we can see that the coefficient estimate is negative,
small and statistically significant at the 1% level. This suggests that
aspirin could be slightly more effective in reducing the probability of
death of older patients (AGE) and of those with high blood pressure
(RSBP). In addition, the mild negative relation of IATE with the delay
between the stroke and randomization (RDELAY) suggests that aspirin
reduces to a greater extent the probability of death of patients that
are hospitalized many hours after the stroke.

Regression on age \| Variable \| coeff \| se \| t-value \| p-value \| \|
--------- \| --------- \| -------- \| ---------- \| ------- \| \|
intercept \| 0.036285 \| 0.003012 \| 12.046842 \| 0.000 \| \| age \|
-0.000495 \| 0.000041 \| -11.957766 \| 0.000 \|

Regression on rsbp \| Variable \| coeff \| se \| t-value \| p-value \|
\| --------- \| --------- \| -------- \| --------- \| ------- \| \|
intercept \| 0.022625 \| 0.002819 \| 8.026694 \| 0.000 \| \| rsbp \|
-0.000137 \| 0.000017 \| -7.883954 \| 0.000 \|

Regression on rdelay \| Variable \| coeff \| se \| t-value \| p-value \|
\| --------- \| --------- \| -------- \| ---------- \| ------- \| \|
intercept \| 0.015143 \| 0.000903 \| 16.771222 \| 0.000 \| \| rdelay \|
-0.000719 \| 0.000038 \| -18.801620 \| 0.000 \|

Table 7: Results of simple linear regressions of IATE against the
continuous variables

To dig deeper into the effects heterogeneity I investigated the
correlates of the extreme values of IATE. To do this I isolated the
observations below the 5th percentile (greatest beneficiaries with IATE
 \< −0.106) and above the 95th percentile (those harmed the most, with
IATE  \> 0.111). Table
<a href="#tab:dummies_comp_table" data-reference-type="ref"
data-reference="tab:dummies_comp_table">8</a> reports the means of the
dummy variables in the top and bottom tails of the IATE distribution.
The same data are visualized in the bar plot in Figure
<a href="#fig:means_dummies_top_vs_bottom" data-reference-type="ref"
data-reference="fig:means_dummies_top_vs_bottom">5</a>. The first
difference to draw attention is that while only 56% of those in the
bottom tail (greatest beneficiaries) had undergone a CT scan (RCT), 85%
of those in the top one (harmed the most) did. This is a remarkable
difference if compared to the 67% of the entire dataset. Analogously,
while in the bottom tail the infarct is visible on a CT scan (RVISINF)
in 43% of the cases, in the top tail this is the case only 29% of the
times. However, a CT scan and the fact that infarct can be seen in it
cannot directly influence the effect of aspirin. Apart from these
variables, a relevant source of heterogeneity between the extreme cases
of IATE seems to be the presence of atrial fibrillation (RATRIAL),
observed in 17% of the patients in the bottom tail against a 29% of
those in the top tail.

| Dummy Variable | Bottom 5% mean | Top 5% mean | Delta (descending) |
|----------------|----------------|-------------|--------------------|
| RCT            | 0.56           | 0.85        | 0.29               |
| RATRIAL        | 0.17           | 0.35        | 0.18               |
| RDEF1          | 0.73           | 0.79        | 0.06               |
| SEX            | 0.46           | 0.51        | 0.05               |
| RDEF4          | 0.48           | 0.52        | 0.04               |
| RDEF5          | 0.21           | 0.24        | 0.03               |
| RDEF6          | 0.23           | 0.25        | 0.02               |
| RDEF2          | 0.86           | 0.88        | 0.02               |
| RASP3          | 0.21           | 0.22        | 0.01               |
| STYPE_POCS     | 0.11           | 0.11        | 0.00               |
| STYPE_OTH      | 0.00           | 0.00        | -0.00              |
| RDEF7          | 0.12           | 0.12        | -0.00              |
| RDEF3          | 0.78           | 0.77        | -0.01              |
| STYPE_PACS     | 0.42           | 0.41        | -0.01              |
| RDEF8          | 0.08           | 0.07        | -0.02              |
| STYPE_LACS     | 0.20           | 0.17        | -0.03              |
| RVISINF        | 0.43           | 0.29        | -0.14              |

Table 8: Means of dummy variables in top vs bottom 5% of the IATE
distribution

The bar chart in Figure
<a href="#fig:rconsc_top_vs_bottom" data-reference-type="ref"
data-reference="fig:rconsc_top_vs_bottom">6</a> compares the percentage
of patients in the three consciousness states (unconscious, drowsy, and
fully alert) in the bottom and the top tails of IATEs. From it we can
see that low IATE scores are associated with a “fully alert” state (86%
in the bottom vs 34% top tail), whereas high IATE scores see a majority
of observations in the “drowsy” and "unconscious" states. This suggests
that aspirin can be more effective in reducing the risk of death when
the patient is completely conscious.

Lastly, I compared the distributions of the continuous variables between
the two tails. Figure <a href="#fig:age" data-reference-type="ref"
data-reference="fig:age">7</a> shows the histograms of AGE in the two
groups. We can see that the trend emerged in Figure
<a href="#fig:cont_vars_scatters" data-reference-type="ref"
data-reference="fig:cont_vars_scatters">4</a> is confirmed: patients
younger than 75 are more likely to end up in the right tail of the IATE
distribution. This is evidence that the treatment might increase risks
for relatively younger people. Figure
<a href="#fig:rsbp" data-reference-type="ref"
data-reference="fig:rsbp">8</a> shows that while for blood pressure
levels (RSBP) lower than 200 the frequencies of high and low IATEs are
comparable, after 200 it is more likely to observe low IATEs. This
suggests that aspirin can be more effective in cases of hypertensive
crisis, when patients have blood pressure levels in excess of 200.
Finally, Figure <a href="#fig:rdelay" data-reference-type="ref"
data-reference="fig:rdelay">9</a> shows that while the frequency of low
IATEs is relatively stable across the values of RDELAY, the frequency of
high IATEs is high for low values of RDELAY ( \< 10 hours) and then
decreases above 30 hours. Thus, the negative trend observed in Figure
<a href="#fig:cont_vars_scatters" data-reference-type="ref"
data-reference="fig:cont_vars_scatters">4</a> and Table
<a href="#tab:reg_IATE_cont_vars" data-reference-type="ref"
data-reference="tab:reg_IATE_cont_vars">7</a> might be due to an adverse
effect of aspirin in the immediate few hours after the stroke.

In light of these findings, I would suggest to treat with aspirin
conscious patients, especially if an hypertensive crisis with no atrial
fibrillation is observed, because in these cases a high reduction of the
probability of death is more likely. Moreover, I would refrain from
treating very young patients, which face a greater risk of negative
effects, and from treating patient very soon after the stroke, which is
associated with a higher likelihood of adverse effects.
