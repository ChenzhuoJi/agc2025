from experiment import Experiment, testExperiment, searchExperiment
import numpy as np

targets = np.load(r"data\interimediate\lasftm_asia\targets.npy")
r = len(np.unique(targets))

params = (2,2,0.5,2)
epr = Experiment("lasftm_asia", r, *params)
epr.run(pred_method = 'kmean')
