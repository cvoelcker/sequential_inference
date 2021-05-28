from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.experiments.factory import setup_experiment
import hydra

from sequential_inference.setup.vector import vector_preface


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    run_dir, preempted = vector_preface(cfg)

    env: gym.Env = setup_environment()
    experiment: AbstractExperiment = setup_experiment(cfg)

    # here I have to decvide where initial data gathering is handeled?
    # I could actually make this a mixin, but then I need to figure
    # out where to handle the vector logic properly (within the data
    # gathering plugin?) This is important to properly handle both
    # OnPolicy and OffPolicy algorithms, since they might behave
    # differently here

    # I could actually try to handle on policy and offpolicy as a mixin,
    # which feels sup[er weird, but the main difference is data
    # gathering strategies, so it could strangely work out pretty well.
    # Essentially we have three Mixin versions: fixed data, buffered
    # data and on-policy data

    # if I do that, I need to hook the reload handler into the mixin
    # hierarchy to essentially prevent that the data is resampled. This
    # sounds strange, but should actually work pretty well, since it is
    # just a version of the dynamic dataset mixin (since the other
    # mixins are pretty much fixed anyways, unless I do a gathering with
    # fixed policy)
