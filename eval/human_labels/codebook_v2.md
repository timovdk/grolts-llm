# Codebook — rating studies against the revised GRoLTS checklist

## What to do

1. Open the workbook in your package. One row per study; `paper_id` matches the file `papers/<paper_id>.pdf` (or `papers/<dataset>/<paper_id>.pdf` if you have two datasets).

1. Score every item **1** if the criterion is reported, **0** if it is not. The cells only accept 0 or 1. Leave nothing blank.

1. Use the two columns before the items as you go — they stay visible while you scroll. **Ambiguous items**: the item numbers you found genuinely unclear for this study, e.g. `7, 13`. **Notes**: why, in your own words. These are data, not admin — which items trained raters find ambiguous is one of the results.

1. Work **blind**: do not look at another rater's sheet, at any existing ratings of these studies, or at any model output, until your sheet is complete. The whole point is an independent estimate of how far two people agree, and it is void if the ratings are anchored on an existing set.

1. Send back the workbook. You do not need to rename it.

## Scoring conventions

These follow the instructions the language models were given, so that the human and model judgements answer the same question:

- Judge **reporting, not quality**. The question is whether the information is present, not whether the choice it describes was a good one.

- Treat references to supplements, appendices, data repositories or URLs mentioned in the text as valid and available — score 1 without chasing the link.

- Score 1 only on **explicit evidence**. If the information is missing, unclear, or merely implied by common practice in the field, score 0.

- Where an item says **“all the models tested”** it means every model that was fitted, not just the selected one; where it says **“the final model”**, only the selected solution matters. Items 11 and 12, and items 13 and 14, are deliberately separated on exactly this distinction. Read those four carefully — it was the largest single source of disagreement in the original checklist.

- If covariates or predictors were not used at all, item 8 is scored 0.

## Items

### Item 1  ·  column D

> Is the metric or unit of time used in the statistical model reported (i.e., wave, hours, days, weeks, months, years, etc.)?

### Item 2  ·  column E

> Is information presented about the mean and variance of time within a wave?

### Item 3  ·  column F

> Is a description provided of how missing data in the analyses were dealt with (i.e., List wise deletion, multiple imputation, Full information maximum likelihood (FIML) etc.)?

### Item 4  ·  column G

> Is information about the distribution of the observed variables included (i.e., tests for normally distributed variables within classes, multivariarte normality, etc.)?

### Item 5  ·  column H

> Is the software that was used for the statistical analysis mentioned?

### Item 6  ·  column I

> Are alternative specifications of within-class heterogeneity considered (e.g., LCGA vs. LGMM) and clearly documented?

### Item 7  ·  column J

> Are alternative shapes/functional forms of the trajectories described (e.g., was it tested whether a quadratic trend or a non-linear form would fit the data better)?

### Item 8  ·  column K

> If covariates or predictors have been used, is it done in such a way that the analyses could be replicated?

### Item 9  ·  column L

> Is information reported about the number of random start values and final iterations included?

### Item 10  ·  column M

> Are the model comparison (and selection) tools described from a statistical perspective?

### Item 11  ·  column N

> Are the total number of fitted models reported?

### Item 12  ·  column O

> Is information about a one-class solution reported?

### Item 13  ·  column P

> Are the number of cases per class reported for the final model (absolute sample size, or proportion)?

### Item 14  ·  column Q

> Are the number of cases per class reported for all the models tested (absolute sample size, or proportion)?

### Item 15  ·  column R

> Is entropy reported?

### Item 16  ·  column S

> Is a plot included with the estimated mean trajectories of the final solution?

### Item 17  ·  column T

> Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?

### Item 18  ·  column U

> Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?

### Item 19  ·  column V

> Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?
