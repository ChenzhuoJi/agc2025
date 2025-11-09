from src.experiment import Experiment, testExperiment, searchExperiment

epr = searchExperiment("citeseer",sample_size=None)
epr.run(pred_method="commutitude")
