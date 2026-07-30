"""Height command generator.

Faithful port of the IsaacLab ``HeightCommand`` reference to mjlab.
Key differences from the IsaacLab original:
  - IsaacLab's ``sensor.data.ray_hits_w`` is replaced by mjlab's
    ``sensor.data.hit_pos_w`` (shape [B, N, 3] vs IsaacLab's [N, 3]).
  - IsaacLab's ``root_pos_w`` (link origin) is replaced by mjlab's explicit
    ``root_link_pos_w``. See #TODO markers.
  - IsaacLab used ``SceneEntityCfg`` to carry the sensor name; mjlab uses a
    plain ``sensor_name: str`` field (cleaner, since SceneEntityCfg is for
    entity components, not sensors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

  from .commands_cfg import HeightCommandCfg


class HeightCommand(CommandTerm):
  r"""Command generator that generates a height command.

  The command consists of the robot's base height.
  """

  cfg: "HeightCommandCfg"
  """The configuration of the command generator."""

  def __init__(self, cfg: "HeightCommandCfg", env: ManagerBasedRlEnv):
    """Initialize the command generator.

    Args:
      cfg: The configuration of the command generator.
      env: The environment.
    """
    super().__init__(cfg, env)

    # obtain the robot entity
    self.robot: Entity = env.scene[cfg.entity_name]
    # sensor name (empty string => flat terrain, no raycast adjustment)
    self.sensor_name = cfg.sensor_name

    # create buffers to store the command
    # -- command: z height
    self.height_command_z = torch.zeros(self.num_envs, 1, device=self.device)

    # metric for height error
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  def __str__(self) -> str:
    """Return a string representation of the command generator."""
    msg = "HeightCommand:\n"
    msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
    msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
    msg += f"\tHeight range: {self.cfg.ranges}"
    return msg

  """
  Properties
  """

  @property
  def command(self) -> torch.Tensor:
    """The desired base height command. Shape is (num_envs, 1)."""
    return self.height_command_z

  """
  Implementation specific functions.
  """

  def _update_metrics(self):
    # time for which the command was executed
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    # logs data
    if self.sensor_name:
      sensor: RayCastSensor = self._env.scene[self.sensor_name]
      # Adjust the target height using the sensor data.
      # IsaacLab: sensor.data.ray_hits_w[..., 2] (shape [N, 3])
      # mjlab:    sensor.data.hit_pos_w       (shape [B, N, 3])
      # Taking [..., 2] then mean over dim=1 yields [B] in both cases.
      adjusted_target_height = self.height_command_z.squeeze(-1) + torch.mean(
        sensor.data.hit_pos_w[..., 2], dim=1
      )
    else:
      # Use the provided target height directly for flat terrain
      adjusted_target_height = self.height_command_z.squeeze(-1)
    # TODO(IsaacLab->mjlab): IsaacLab root_pos_w is at link origin.
    # mjlab makes this explicit: root_link_pos_w.
    self.metrics["error_height"] += (
      torch.abs(adjusted_target_height - self.robot.data.root_link_pos_w[:, 2]) / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor):
    # sample height commands
    r = torch.empty(len(env_ids), device=self.device)
    self.height_command_z[env_ids, 0] = r.uniform_(*self.cfg.ranges.height_z)

  def _update_command(self):
    """Post-processes the height command."""
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # No debug visualization for the height command (IsaacLab reference was a no-op).
    pass
