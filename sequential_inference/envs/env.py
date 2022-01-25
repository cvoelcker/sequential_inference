from sequential_inference.envs.vec_env.vec_wrapper import VecTimeLimitWrapper
from sequential_inference.envs.vec_env.vec_torch import VecTorch
from sequential_inference.envs.meta_wrappers.single_task import SingleTaskEnv
from sequential_inference.envs.vec_env.vec_env import VecEnv
# from sequential_inference.envs.meta_wrappers.meta_world_wrappers import ML1Env
from sequential_inference.envs.meta_wrappers.mujoco_meta_tasks import MujocoMetaTask
from sequential_inference.envs.vec_env import SubprocVecEnv, DummyVecEnv
from sequential_inference.envs.meta_wrappers.wipe_meta import WipeMeta
from sequential_inference.envs.vec_env.vec_normalize import VecNormalize


def make_multi_env(env: str, suite: str) -> VecEnv:
    if suite == "ml1":
        env = ML1Env(task_name=env)
    elif suite == "mujoco":
        env = MujocoMetaTask(env)
    elif suite == "wipe":
        env = WipeMeta(env)
    else:
        raise NotImplementedError("benchmark {}".format(suite))
    return env


def make_vec_env(
    n_envs: int,
    env_name: str,
    time_limit: int = -1,
    is_multi_env: bool = False,
    normalize_rew: bool = False,
    ret_rms: bool = False,
    suite: str = None,
) -> VecEnv:
    if is_multi_env:
        assert suite is not None, "Can only create multi env from known suite"
        env_func = lambda: make_multi_env(env_name, suite)
    else:
        env_func = lambda: SingleTaskEnv(env_name)

    if n_envs > 1:
        venvs = SubprocVecEnv([env_func for _ in range(n_envs)])
    else:
        venvs = DummyVecEnv([env_func])
    if normalize_rew:
        venvs = VecNormalize(venvs, normalise_rew=normalize_rew, ret_rms=ret_rms)

    if time_limit >= 1:
        venvs = VecTimeLimitWrapper(venvs, duration=time_limit)
    return VecTorch(venvs)


def setup_environment(cfg):
    env_name = cfg.env.env_name
    n_envs = cfg.env.n_envs
    time_limit = cfg.env.time_limit
    is_multi_env = cfg.env.is_multi_env
    normalize_rew = cfg.algorithm.normalize_rew
    ret_rms = cfg.algorithm.ret_rms
    suite = cfg.env.suite

    return make_vec_env(
        n_envs, env_name, time_limit, is_multi_env, normalize_rew, ret_rms, suite
    )
