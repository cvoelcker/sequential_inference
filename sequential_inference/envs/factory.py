from sequential_inference.envs.meta_wrappers.common import (
    VisualTaskWrapper,
    TimeLimitWrapper,
)
from sequential_inference.envs.meta_wrappers.meta_world_wrappers import ML1Env
from sequential_inference.envs.meta_wrappers.mujoco_meta_tasks import MujocoMetaTask
from sequential_inference.envs.vec_env import SubprocVecEnv, DummyVecEnv
from sequential_inference.envs.meta_wrappers.wipe_meta import WipeMeta
from sequential_inference.envs.vec_env.vec_normalize import VecNormalize


def make_env(is_eval_env, env_cfg, single_task=False):
    if env_cfg.benchmark == "ml1":
        env = ML1Env(task_name=env_cfg.task, is_eval_env=is_eval_env)
        env = TimeLimitWrapper(env, duration=env_cfg.horizon)
    elif env_cfg.benchmark == "mujoco":
        env = MujocoMetaTask(
            env_cfg.task,
            is_eval_env=is_eval_env,
            reward_scaling=env_cfg.reward_scaling,
            single_task=single_task,
            distance=env_cfg.distance,
        )
    elif env_cfg.benchmark == "wipe":
        env = WipeMeta(
            task_code=env_cfg.task,
            num_paths=env_cfg.num_paths,
            num_markers=env_cfg.num_markers,
            is_eval_task=False,
        )
        env = TimeLimitWrapper(env, duration=env_cfg.horizon)
    else:
        raise NotImplementedError("benchmark {}".format(env_cfg.benchmark))
    if env_cfg.visual_obs:
        env = VisualTaskWrapper(env, resolution=env_cfg.resolution)
    return env


def make_vec_env(
    n_envs,
    is_eval_env,
    env_cfg,
    normalize=False,
    normalize_rew=None,
    ret_rms=None,
    single_task=False,
):
    print("Running subproc vec env")
    if n_envs > 1:
        venvs = SubprocVecEnv(
            [
                lambda: make_env(is_eval_env, env_cfg, single_task=single_task)
                for _ in range(n_envs)
            ]
        )
        print(venvs)
    else:
        venvs = DummyVecEnv(
            [lambda: make_env(is_eval_env, env_cfg, single_task=single_task)]
        )
    if normalize:
        venvs = VecNormalize(venvs, normalise_rew=normalize_rew, ret_rms=ret_rms)
    return venvs
