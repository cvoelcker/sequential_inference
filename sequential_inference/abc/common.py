import abc
import pickle
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Tuple,
    TypeVar,
    Union,
)
from gym import Space

import torch


class Env(abc.ABC):
    num_envs: int
    action_space: Space
    observation_space: Space

    @abc.abstractmethod
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abc.abstractmethod
    def reset(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def render(self, mode: str):
        pass

    @abc.abstractmethod
    def reset_task(self, task_id: Union[int, List[int]]):
        pass


class Saveable:
    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__.update(pickle.load(f))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, 2)


class Checkpointable(abc.ABC):
    """
    An abstract class for components that can be saved and loaded.

    The components are kept in a nested dictionary, where the leaves are pytorch modules.
    """

    device: str

    def __init__(self):
        self.device = "cpu"
        self.model_buffer: Dict[
            str,
            Union[Checkpointable, torch.nn.Module, torch.Tensor, torch.optim.Optimizer],
        ] = {}
        super().__init__()

    def state_dict(self):
        """Returns the state dict from the model buffer (compatibility with torch.nn.Module)

        Returns:
            Dict: the (potentially nested) state dict
        """
        to_save = {}
        for k, v in self.model_buffer.items():
            if isinstance(v, torch.Tensor):
                to_save[k] = v
            else:
                to_save[k] = v.state_dict()
        return to_save

    def load_state_dict(self, chp: Dict[str, OrderedDict]):
        """Loads the dictionary into the model buffer (compatibility with torch.nn.Module)

        Args:
            chp (Dict[str, OrderedDict]): the (potentially nested) state dict that should be loaded
        """
        for k, v in self.model_buffer.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = chp[k]
            else:
                v.load_state_dict(chp[k])

    def load(self, directory: str):
        """
        Loads the state dict from the given directory and loads it into the model buffer.
        """
        chp = torch.load(directory, map_location=self.device)
        self.load_state_dict(chp)

    def register_module(
        self,
        key: str,
        module: "Union[Checkpointable, torch.nn.Module, torch.optim.Optimizer]",
    ):
        """
        Registers a module in the model buffer.
        Args:
            key (str): the key to register the module under
            module (torch.nn.Module or Dict): the module to register (needs to provide state dict)
        """

        if key in self.model_buffer.keys():
            raise KeyError("Key in module buffer is not unique")
        else:
            self.model_buffer[key] = module

    def to(self, device: str):
        self.device = device
        for _, v in self.model_buffer.items():
            if not isinstance(v, torch.optim.Optimizer):
                v.to(device)

    def summarize(self):
        from torchinfo import summary

        for k, v in self.model_buffer.items():
            if isinstance(v, torch.nn.Module):
                print("++++++++++++++++++++++++++++++++++++++")
                print(f"++++++ {k}")
                print("++++++++++++++++++++++++++++++++++++++")
                summary(v)
                print()
            elif isinstance(v, Checkpointable):
                v.summarize()


class TorchContainer(Checkpointable):
    def load_state_dict(self, chp: Dict[str, OrderedDict]):
        for k, v in chp.items():
            self.__dict__[k] = v


class AbstractAlgorithm(Checkpointable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Dict]:
        """
        Abstract method to compute a loss function from an observation sequence

        Generally expects the tensors to be of shape (batch_size, horizon, ...) and indexed using regular RL notation.

        Args:
            obs (torch.Tensor): a tensor of observations
            actions (Optional[torch.Tensor], optional): a tensor of actions. Defaults to None.
            rewards (Optional[torch.Tensor], optional): a tensor of rewards. Defaults to None.
            done (Optional[torch.Tensor], optional): a tensor of done flags. Defaults to None.

        Raises:
            NotImplementedError: cannot be called directly, needs to be subclassed

        Returns:
            torch.Tensor: the loss
            Dict: a dictionary containing additional information about the loss
        """
        raise NotImplementedError("Cannot instantiate Abstract")

    def get_optimizer(self) -> Union[torch.optim.Optimizer, Dict]:
        """Returns the optimizer for the contained modules

        Returns:
            torch.optim.Optimizer: the optimizer or a dictionary of optimizers
        """
        raise NotImplementedError("Cannot instantiate Abstract")

    @abc.abstractmethod
    def get_step(self) -> Callable:
        pass
