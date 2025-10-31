import numpy as np

from src.experiment import Experiment, testExperiment, searchExperiment

epr = Experiment("lasftm_asia",2,5,0.2,0.2,100)
epr.run(pred_method="laplace")
epr.write_experiment_log()
