# Educational Achievement Case Study

## Search Strategy and Data Retrieval
To identify studies applying latent trajectory modeling to educational outcomes, the OpenAlex database was queried. The search strategy targeted publications containing the terms “Latent Growth Mixture Model” or “Latent Class Growth Analysis” in the full text, combined with the term “achievement” in the title or abstract.

### Search string
```
("latent growth mixture model" OR "latent class growth analysis")
AND achievement
```
This search yielded 371 articles, of which 259 (69.8%) were open access. Access to the remaining articles was ensured via the Utrecht University Library browser extension. The retrieved records were exported with title, abstract, and DOI information for screening.
Although the search string did not explicitly restrict results to the educational domain (e.g., “educational achievement”), domain relevance was determined during the screening stage.

## Screening Procedure
The dataset of 371 articles was imported into ASReview (version 1.6) for AI-assisted screening. Default model settings were used:
- Feature extraction: TF-IDF
- Classifier: Naïve Bayes
- Query strategy: Maximum
- Balance strategy: Dynamic Resampling (Double)

To initialize the active learning model, 16 seed articles were labeled during the warm-up phase: 10 relevant and 6 irrelevant.
No stopping rule was predefined due to unfamiliarity with the software and the intention to ensure that no relevant studies were missed. Therefore, all 371 articles were screened manually based on titles and abstracts. When a record appeared potentially relevant, the full text was accessed via the DOI link for additional verification.
During screening, ASReview dynamically reordered the dataset by prioritizing studies predicted to be relevant based on the reviewer’s labeling decisions. The analytics output indicated that all relevant studies were identified within the first 140 screened articles (39%), demonstrating the efficiency of the active learning approach.

## Eligibility Criteria
Articles were considered relevant if they:

- Applied Latent Growth Mixture Modeling (LGMM) or Latent Class Growth Analysis (LCGA).
- Investigated achievement-related outcomes within an educational context.

Articles using other latent trajectory approaches or focusing on non-achievement-related outcomes were excluded.

## Final Sample
After screening, 32 articles met the eligibility criteria and were included in the final dataset.

## Checklist Scoring Procedure
The included articles were evaluated using the revised GRoLTS checklist. A custom scoring spreadsheet was created in Microsoft Excel to record the results.
Checklist items were scored as:
- 1 = criterion included
- 0 = criterion not included

Initially, approximately two articles were reviewed per day. The time required per article decreased from over one hour during the first assessments to approximately 20 minutes as familiarity with the checklist increased. In total, the manual scoring process required approximately 20 hours.