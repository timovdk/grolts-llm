p1 = {
    0: "Is the metric of time used in the statistical model reported?",
    1: "Is information presented about the mean and variance of time within a wave?",
    2: "Is the missing data mechanism reported?",
    3: "Is a description provided of what variables are related to attrition/missing data?",
    4: "Is a description provided of how missing data in the analyses were dealt with?",
    5: "Is information about the distribution of the observed variables included?",
    6: "Is the software mentioned?",
    7: "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    8: "Are alternative specifications of the between-class differences in variancecovariance matrix structure considered and clearly documented?",
    9: "Are alternative shape/functional forms of the trajectories described?",
    10: "If covariates have been used, can analyses still be replicated?",
    11: "Is information reported about the number of random start values and final iterations included?",
    12: "Are the model comparison (and selection) tools described from a statistical perspective?",
    13: "Are the total number of fitted models reported, including a one-class solution?",
    14: "Are the number of cases per class reported for each model (absolute sample size, or proportion)?",
    15: "If classification of cases in a trajectory is the goal, is entropy reported?",
    16: "Is a plot included with the estimated mean trajectories of the final solution?",
    17: "Are plots included with the estimated mean trajectories for each model?",
    18: "Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?",
    19: "Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    20: "Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?",
}

p2 = {
    0: "Is the metric or unit of time used in the statistical model reported?",
    1: "Is information presented about the mean and variance of time within a wave?",
    2: "Is the missing data mechanism reported?",
    3: "Is a description provided of what variables are related to attrition/missing data?",
    4: "Is a description provided of how missing data in the analyses were dealt with?",
    5: "Is information about the distribution of the observed variables included?",
    6: "Is the software that was used for the statistical analysis mentioned?",
    7: "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    8: "Are alternative specifications of the between-class differences in variancecovariance matrix structure considered and clearly documented?",
    9: "Are alternative shape/functional forms of the trajectories described?",
    10: "If covariates or predictors have been used, is it done in such a way that the analyses could be replicated?",
    11: "Is information reported about the number of random start values and final iterations included?",
    12: "Are the model comparison (and selection) tools described from a statistical perspective?",
    13: "Are the total number of fitted models reported, including a one-class solution?",
    14: "Are the number of cases per class reported for each model (absolute sample size, or proportion)?",
    15: "If classification of cases in a trajectory is the goal, is entropy reported?",
    16: "Is a plot included with the estimated mean trajectories of the final solution?",
    17: "Are plots included with the estimated mean trajectories for each model?",
    18: "Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?",
    19: "Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    20: "Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?",
}

p3 = {
    0: "Is the metric or unit of time used in the statistical model reported? (i.e., hours, days, weeks, months, years, etc.)",
    1: "Is information presented about the mean and variance of time within a wave?(mean and variance of: within measurement occasion, mean and variance of: within a period of time, etc.)",
    2: "Is the missing data mechanism reported? (i.e., missing at random (MAR), Missing not at random (MNAR), missing completely at random (MCAR), etc.) ",
    3: "Is a description provided of what variables are related to attrition/missing data? (i.e., a dropout effect, auxillary variables, skip patterns, etc.)",
    4: "Is a description provided of how missing data in the analyses were dealt with?(i.e., List wise deletion, multiple imputation, Full information maximum likelihood (FIML) etc.)",
    5: "Is information about the distribution of the observed variables included? (i.e., tests for normally distributed variables within classes, multivariarte normality, etc.) ",
    6: "Is the software that was used for the statistical analysis mentioned? (i.e., Mplus, R, etc.)",
    7: "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    8: "Are alternative specifications of the between-class differences in variance covariance matrix structure considered and clearly documented? (i.e., constrained accros subgroups, fixed accross subgroups, etc.)",
    9: "Are alternative shape/functional forms of the trajectories described? (e.g., was it tested whether a quadratic trend or a non-linear form would fit the data better)",
    10: "If covariates or predictors have been used, is it done in such a way that the analyses could be replicated? (e.g., was it reported they used time-varying or time-invariant covariates at the level of the dependent or independent variables)",
    11: "Is information reported about the number of random start values and final iterations included? (e.g., If ML has been used to estimate the latent trajectory model, then it should be reported if the final class solution has converged to the maximum of the ML distribution and not on a local maxima.)",
    12: "Are the model comparison (and selection) tools described from a statistical perspective? (i.e., BIC, AIC, etc.)",
    13: "Are the total number of fitted models reported, including a one-class solution?",
    14: "Are the number of cases per class reported for each model (absolute sample size, sample size per class or proportion)?",
    15: "If classification of cases in a trajectory is the goal, is entropy reported? (i.e., the relative entropy value, the number of misclassifications per model)",
    16: "Is a plot included with the estimated mean trajectories of the final solution?",
    17: "Are plots included with the estimated mean trajectories for each model?",
    18: "Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?",
    19: "Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    20: "Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?",
}

## Update these questions ##
p4 = {
    0: "Is the metric or unit of time used in the statistical model reported? (i.e., wave, hours, days, weeks, months, years, etc.)",
    1: "Is information presented about the mean and variance of time within a wave?",
    2: "Is a description provided of how missing data in the analyses were dealt with?(i.e., List wise deletion, multiple imputation, FIML etc.)",
    3: "Is the estimator for the analysis reported? ",
    4: "Is the software that was used for the statistical analysis mentioned?",
    5: "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    6: "Are alternative shape/functional forms of the trajectories described? (e.g., was it tested whether a quadratic trend or a non-linear form would fit the data better)",
    7: "If covariates or predictors have been used, is it done in such a way that the analyses could be replicated",
    8: "Is information reported about the number of random start values and final iterations included?",
    9: "Are the model comparison (and selection) tools described from a statistical perspective?",
    10: "Is there a table describing the model fit of all models evaluated?",
    11: "Is information about a one-class solution reported?",
    12: "Are the number of cases per class reported for the final model (absolute sample size, or proportion)?",
    13: "Are the number of cases per class reported for the all models tested (absolute sample size, or proportion)?",
    14: "Is entropy reported?",
    15: "Is a plot included with the estimated mean trajectories of the final solution?",
    16: "Are descriptives of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    17: "Are the data or code files available (either in the appendix, supplementary or online materials)?"
}


def get_questions(experiment_id):
    if experiment_id == 0:
        return p1
    elif experiment_id == 1:
        return p2
    elif experiment_id == 2:
        return p3
    elif experiment_id == 3:
        return p4
    else:
        print("ERROR: No questions defined")
        exit(1)
