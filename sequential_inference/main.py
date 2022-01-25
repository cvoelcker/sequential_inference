import hydra

from sequential_inference.abc.env import Env
from sequential_inference.abc.common import AbstractAlgorithm
from sequential_inference.abc.experiment import AbstractExperiment

from sequential_inference.setup.vector import vector_preface
from sequential_inference.envs.env import setup_environment
from sequential_inference.setup.factory import setup_experiment
from sequential_inference.setup.logging import setup_logging


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    run_dir, preempted = vector_preface(cfg)

    env: Env = setup_environment(cfg)
    experiment: AbstractExperiment = setup_experiment(cfg)
    logging = setup_logging(cfg)

    experiment.register_observers(logging)
    
    # nothing should really run before here, no data loading etc to prevent issues on server with interrupt
    # since we might have separate training stages, we need to handle that well somewhere
    experiment.build(run_dir, preempted)
    experiment.run()