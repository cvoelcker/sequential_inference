"""
Adapted from https://github.com/openai/baselines/
"""
import numpy as np
from sequential_inference.envs.vec_env.vec_env import VecEnv


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.task_size = env.task_size
        self.task_classify = env.task_classify

        obs_dtype = np.uint8 if len(env.observation_space.shape) > 1 else np.float32
        self.buf_obs = np.zeros(
            (self.num_envs, *env.observation_space.shape), dtype=obs_dtype
        )
        self.buf_rews = np.zeros((self.num_envs, 1), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert (
                self.num_envs == 1
            ), "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs
            )
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            (
                self.buf_obs[e],
                self.buf_rews[e][0],
                self.buf_dones[e],
                self.buf_infos[e],
            ) = self.envs[e].step(action)
            # if self.buf_dones[e]:
            #     self.buf_obs[e] = self.envs[e].reset()
            # self._save_obs(e, obs)
        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            self.buf_infos.copy(),
        )

    def reset(self):
        for e in range(self.num_envs):
            self.buf_obs[e] = self.envs[e].reset()
        return np.copy(self.buf_obs)

    def get_images(self, resolution):
        return [env.render(resolution) for env in self.envs]

    def close_extras(self):
        for e in self.envs:
            e.close()

    def get_task(self):
        return np.stack([env.get_task() for env in self.envs])

    # def render(self, mode='human'):
    #     if self.num_envs == 1:
    #         return self.envs[0].render(mode=mode)
    #     else:
    #         return super().render(mode=mode)

    # def _save_obs(self, e, obs):
    #     for k in self.keys:
    #         if k is None:
    #             self.buf_obs[k][e] = obs
    #         else:
    #             self.buf_obs[k][e] = obs[k]
