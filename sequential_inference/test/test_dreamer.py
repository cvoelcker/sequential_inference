import hydra
import torch

from sequential_inference.algorithms.algorithm_factory import make_sequence_algorithm
from sequential_inference.trainers.trainer import ModelTrainer


@hydra.main(config_path="../../config", config_name="tests/dreamer_test")
def main(cfg):
    print(cfg)
    algorithm = make_sequence_algorithm(3, 3, 12, cfg)
    algorithm.to("cpu")
    state_dict = algorithm.get_checkpoint()
    algorithm.load_checkpoint(state_dict)
    print(algorithm.get_parameters())
    print(sum([t.numel() for t in algorithm.get_parameters()]))

    trainer = ModelTrainer(algorithm)


if __name__ == "__main__":
    main()
