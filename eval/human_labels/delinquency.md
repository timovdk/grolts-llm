# Adolescent Delinquency Case Study

## Search Strategy and Data Retrieval
To construct the initial dataset of relevant publications, the OpenAlex database was queried using its API. A Python script executed in a Jupyter Notebook was used to programmatically retrieve records and export them to a CSV file containing titles, abstracts, and DOI links.
The search targeted studies applying latent trajectory modeling techniques to delinquent or externalizing behavior.

### Search string
```
(delinquency OR delinquent OR criminal OR offender OR violence OR aggression)
AND ("latent growth mixture" OR "latent class growth" OR LCGA OR LGMM)
```
In addition, results were filtered to include publications from 2015 onward, ensuring the inclusion of relatively recent studies following the increasing adoption of latent trajectory models and the introduction of the GRoLTS checklist in 2017.
The query yielded 407 records, which were exported to CSV format for screening in ASReview.

## Screening Procedure
The dataset was imported into ASReview, an open-source tool for AI-assisted systematic reviewing that uses an active learning cycle to prioritize potentially relevant studies.
During screening, the reviewer labeled records as relevant or irrelevant based on their abstracts and, where necessary, the methods section of the full text. As labeling progressed, the machine learning model updated its predictions and prioritized studies likely to be relevant.
Screening was terminated when the stopping rule of 50 consecutive irrelevant papers was reached. At that point:
- 258 papers had been screened manually.
- 69 papers had been labeled as relevant.

## Eligibility Criteria
Articles were considered relevant if they met the following criteria:

- The study applied LGMM or LCGA methods.
- The sample included individuals aged 13 years or older exhibiting externalizing behavior, such as delinquency, offending, antisocial behavior, deviant behavior, or problematic substance use.
- The study identified distinct trajectory classes related to these behaviors.

Studies using alternative modeling approaches (e.g., Latent Growth Curve Models, Latent Transition Analysis, Longitudinal Latent Class Analysis, Multigroup Latent Growth Curve Modeling) were excluded.

## Final Sample
Among the 69 relevant records identified, four duplicates were detected. In each case, the most recent version of the paper was retained and the older record removed (record IDs: 371, 280, 44, and 162).
This resulted in a final dataset of 65 unique articles.
Due to time constraints, only the first 37 articles (ranked by ASReview relevance) were included in the checklist scoring stage for this project.

## Checklist Scoring Procedure
The included articles were evaluated using the revised GRoLTS checklist. A custom scoring spreadsheet was created in Microsoft Excel to record the results.
Checklist items were scored as:
- 1 = criterion included
- 0 = criterion not included

Initially, approximately two articles were reviewed per day. The time required per article decreased from over one hour during the first assessments to approximately 20 minutes as familiarity with the checklist increased. In total, the manual scoring process required approximately 20 hours.