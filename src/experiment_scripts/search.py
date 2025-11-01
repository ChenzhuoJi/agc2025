from src.experiment import Experiment, testExperiment, searchExperiment

epr = searchExperiment("lasftm_asia", 100)
epr.run(pred_method=None)
