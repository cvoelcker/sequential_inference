from gym.envs.registration import register

# import d4rl

register(
    "HalfCheetahDir-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.mujoco.half_cheetah_dir:HalfCheetahDirEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "HalfCheetahVel-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.mujoco.half_cheetah_vel:HalfCheetahVelEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "HalfCheetah-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.mujoco.half_cheetah:HalfCheetahEnv"
    },
    max_episode_steps=200,
)

register(
    id="SawyerReach-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.reacher.sawyer_reacher:SawyerReachingEnv"
    },
    max_episode_steps=40,
)

register(
    id="SawyerReachGoal-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.reacher.sawyer_reacher:SawyerReachingEnvGoal"
    },
    max_episode_steps=40,
)

register(
    id="SawyerReachGoalOOD-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.reacher.sawyer_reacher:SawyerReachingEnvGoalOOD"
    },
    max_episode_steps=40,
)

register(
    id="SawyerReachGoalTwoModes-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.reacher.sawyer_reacher:SawyerReachingEnvGoalComplexTwoModes"
    },
    max_episode_steps=40,
)

register(
    id="SawyerHammer-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.sawyer.sawyer_hammer:SawyerHammerEnv"
    },
    max_episode_steps=40,
)


register(
    id="SawyerHammerGoal-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.sawyer.sawyer_hammer:SawyerHammerEnvGoal"
    },
    max_episode_steps=40,
)


register(
    id="SawyerDrawer-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.sawyer.sawyer_drawer:SawyerDrawerEnv"
    },
    max_episode_steps=40,
)


register(
    id="SawyerDrawerGoal-v0",
    entry_point="sequential_inference.envs.meta_wrappers.mujoco_meta_tasks:mujoco_wrapper",
    kwargs={
        "entry_point": "sequential_inference.envs.sawyer.sawyer_drawer:SawyerDrawerEnvGoal"
    },
    max_episode_steps=40,
)

#
# register(
#     id='SawyerReachMT-v0',
#     entry_point='envs.reacher.sawyer_reacher:SawyerReachingEnvMultitask',
#     max_episode_steps=40,
# )
