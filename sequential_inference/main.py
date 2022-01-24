from sequential_inference.abc.common import AbstractAlgorithm
from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.experiments.factory import setup_experiment
import hydra

from sequential_inference.setup.vector import vector_preface


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    run_dir, preempted = vector_preface(cfg)

    env: gym.Env = setup_environment(cfg)
    experiment: AbstractExperiment = setup_experiment(cfg)
    logging: AbstractLogging = setup_logging(cfg)

    experiment.set_logging(logging)
    
    # nothing should really run before here, no data loading etc to prevent issues on server with interrupt
    # since we might have separate training stages, we need to handle that well somewhere
    experiment.build(run_dir, preempted)
    experiment.run()