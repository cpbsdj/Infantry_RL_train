from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_env_cfg import infantry_flat_env_cfg
from .rough_env_cfg import infantry_rough_env_cfg
from .rl_cfg import infantry_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Infantry",
  env_cfg=infantry_rough_env_cfg(),
  play_env_cfg=infantry_rough_env_cfg(play=True),
  rl_cfg=infantry_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Infantry",
  env_cfg=infantry_flat_env_cfg(),
  play_env_cfg=infantry_flat_env_cfg(play=True),
  rl_cfg=infantry_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
