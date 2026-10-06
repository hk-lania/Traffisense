# Early prototypes

First versions of the density and queue estimators. They have been replaced by
`metrics/density.py` and `metrics/queue_length.py`, which handle overlapping
boxes, stop lines and stop/go hysteresis. Kept for reference (e.g. to show how the
approach evolved in the report). Nothing in the pipeline imports these files.
