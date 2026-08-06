# System design
[WARNING] Live doc will change.

## Top level breakdown
- define problem (README)
- survey basic algos for it
- explore if there are existing labeled public datasets OR create a synthetic one
- write basic train/test loop
- produce estimator artifact, measure basic goodness
- wire in prod:
  - estimator store (psql JSONB blob should suffice),
  - prediction serving endpoint

## Basic algorithms
### data preprocessing
- normalization
### matching algo
- exact matching
- sthsth about ngrams
- sthsth more complex

