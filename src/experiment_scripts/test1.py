import numpy as np

from src.experiment import Experiment, testExperiment, searchExperiment

epr = Experiment("lasftm_asia", 3, 2, 0.2, 0.2, 100)
epr.run(pred_method="kmeans",write_log=True)
epr.model.loss_tracker.plot_loss()
