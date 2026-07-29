from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg

from infantry.tasks.velocity.config.infantry.rough_env_cfg import (
  infantry_rough_env_cfg,
)

TerrainType = Literal["rough", "obstacles"]


def infantry_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create infantry flat terrain velocity configuration."""
  cfg = infantry_rough_env_cfg(play=play)

  cfg.scene.num_envs = 4096
  cfg.scene.extent = 5.0

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  return cfg
