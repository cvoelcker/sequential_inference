import gym
import torch
import numpy as np

from sequential_inference.envs.vec_env.vec_env import VecEnv, VecEnvWrapper


class VecTorch(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

        if isinstance(venv.observation_space, gym.spaces.Box):
            self.observation_space = TorchBox(
                venv.num_envs, venv.observation_space.low, venv.observation_space.high
            )
        if isinstance(venv.action_space, gym.spaces.Box):
            self.action_space = TorchBox(
                venv.num_envs, venv.action_space.low, venv.action_space.high
            )

    def reset(self):
        obs = self.venv.reset()
        return torch.from_numpy(transpose_numpy_torch_img(obs))

    def reset_mdp(self):
        obs = self.venv.reset_mdp()
        return torch.from_numpy(transpose_numpy_torch_img(obs))

    def step_async(self, actions):
        actions = actions.detach().cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = torch.from_numpy(transpose_numpy_torch_img(obs))
        rews = torch.from_numpy(rews)
        news = torch.from_numpy(news)
        vec_infos = {}
        for k in infos[0].keys():
            vec_infos[k] = torch.stack(
                [
                    torch.from_numpy(e[k])
                    if isinstance(e[k], np.ndarray)
                    else torch.Tensor([e[k]])
                    for e in infos
                ],
                0,
            )
        return obs, rews, news, vec_infos


def transpose_numpy_torch_img(array: np.ndarray) -> np.ndarray:
    if len(array.shape) == 3:
        return np.transpose(array, (2, 0, 1))
    elif len(array.shape) == 4:
        return np.transpose(array, (0, 3, 1, 2))
    else:
        return array


class TorchBox:
    def __init__(self, num_envs: int, low: np.ndarray, high: np.ndarray):
        self.num_envs = num_envs
        assert low.shape == high.shape
        self.low: torch.Tensor = torch.from_numpy(low)
        self.high: torch.Tensor = torch.from_numpy(high)

    def sample(self) -> torch.Tensor:
        unif_sample = torch.rand(self.num_envs, *self.low.shape)
        if torch.all(torch.isfinite(self.low)):
            return (self.high - self.low) * unif_sample + self.low
        else:
            return 2 * unif_sample - 1.0

    @property
    def shape(self):
        return self.low.shape


class VecTimeLimitWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, duration: int):
        super().__init__(venv)
        self.duration = duration
        self.passed_steps = None

    def step_wait(self):
        assert self.passed_steps is not None, "Must reset environment."
        obs, reward, done, info = self.venv.step_wait()
        self.passed_steps += 1
        if self.passed_steps >= self.duration:
            done[:] = True
            self.passed_steps = None
        return obs, reward, done, info

    def reset(self):
        self.passed_steps = 0
        return self.venv.reset()
