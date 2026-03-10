# Root Makefile - delegates to c/Makefile
# Run from project root: make, make run, make results, etc.

.PHONY: all clean run results plots plots-all compare verify regenerate help
.PHONY: threshold_analysis prepare_dataset learning_rate_experiment iterations_experiment outlier_values

all:
	$(MAKE) -C c all

clean:
	$(MAKE) -C c clean

run: all
	$(MAKE) -C c run

results: all
	$(MAKE) -C c results

plots: results
	$(MAKE) -C c plots

plots-all:
	$(MAKE) -C c plots-all

compare: results
	$(MAKE) -C c compare

verify:
	$(MAKE) -C c verify

regenerate:
	$(MAKE) -C c regenerate

help:
	$(MAKE) -C c help

threshold_analysis:
	$(MAKE) -C c threshold_analysis

prepare_dataset:
	$(MAKE) -C c prepare_dataset

learning_rate_experiment:
	$(MAKE) -C c learning_rate_experiment

iterations_experiment:
	$(MAKE) -C c iterations_experiment

outlier_values:
	$(MAKE) -C c outlier_values
