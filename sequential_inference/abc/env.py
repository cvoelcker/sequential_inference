import abc

import gym

class Env(abc.Abc):
    pass

Env.register(gym.Env)
