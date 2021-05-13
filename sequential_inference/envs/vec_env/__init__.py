"""
Adapted from https://github.com/openai/baselines/
"""

from .subproc_vec_env import SubprocVecEnv
from .dummy_vec_env import DummyVecEnv


__all__ = ["SubprocVecEnv", "DummyVecEnv"]
