import abc
import gym


class MetaTask(abc.ABC):
    """
    Abstract base class for all environments
    """

    def reset(self, reset_task=True):
        if reset_task:
            self.reset_task()
        return self.reset_mdp()

    @abc.abstractmethod
    def reset_task(self):
        """
        Reset task in the environment.
        Return:
            None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset_mdp(self):
        """
        Reset MDP without changing the task.
        Returns:
            Initial observation after reset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, resolution):
        """
        Render the current state of enviroment.
        Args:
            resolution: tuple of higth and width of the image
        Return:
            image of shape [resolution[0], resolution[1], C]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Free all the resources
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_task(self):
        """
        Return current task (a vector of discrete or continuous values)
        """
        raise NotImplementedError


class VisualTaskWrapper(gym.ObservationWrapper):
    def __init__(self, env, resolution=(128, 128)):
        super(VisualTaskWrapper, self).__init__(env)
        self.env = env
        self.resolution = resolution

        pixels = self.observation(None)

        self.observation_space = gym.spaces.Box(
            shape=pixels.shape, low=0, high=255, dtype=pixels.dtype
        )
        self.action_space = env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def observation(self, observation):
        """
        Returns rendered image
        """
        return self.env.render(self.resolution)


class TimeLimitWrapper:
    def __init__(self, env, duration):
        self.env = env
        self.duration = duration
        self.passed_steps = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        assert self.passed_steps is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self.passed_steps += 1
        if self.passed_steps >= self.duration:
            done = True
            self.passed_steps = None
        else:
            # for robosute tasks, otherwise they terminates early
            done = False
        return obs, reward, done, info

    def reset(self):
        self.passed_steps = 0
        return self.env.reset()
