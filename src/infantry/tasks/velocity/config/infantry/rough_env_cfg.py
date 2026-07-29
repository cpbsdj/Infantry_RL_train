from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg

from infantry.robot.infantry.infantry_constants import get_infantry_robot_cfg
from infantry.tasks.velocity.config.infantry.env_cfgs import (
  make_infantry_velocity_env_cfg,
)

TerrainType = Literal["rough", "obstacles"]


def infantry_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create infantry rough terrain velocity configuration."""
  cfg = make_infantry_velocity_env_cfg()

  cfg.scene.num_envs = 4096
  cfg.scene.extent = 5.0
  
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_infantry_robot_cfg()}

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
