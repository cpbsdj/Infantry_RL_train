"""Custom metrics for viser GUI display.

These compute per-step command tracking errors as raw physical quantities
(m/s, rad/s, m).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def command_error_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-step linear velocity tracking error [m/s] (xy plane)."""
  cmd = env.command_manager.get_term("base_velocity")
  return torch.norm(
    cmd.vel_command_b[:, :2] - cmd.robot.data.root_com_lin_vel_b[:, :2], dim=-1
  )


def command_error_vel_yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-step angular velocity tracking error [rad/s] (z axis)."""
  cmd = env.command_manager.get_term("base_velocity")
  return torch.abs(
    cmd.vel_command_b[:, 2] - cmd.robot.data.root_com_ang_vel_b[:, 2]
  )


def command_error_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-step base height tracking error [m]."""
  cmd = env.command_manager.get_term("base_height")
  if cmd.sensor_name:
    sensor: RayCastSensor = env.scene[cmd.sensor_name]
    adjusted_target = cmd.height_command_z.squeeze(-1) + torch.mean(
      sensor.data.hit_pos_w[..., 2], dim=1
    )
  else:
    adjusted_target = cmd.height_command_z.squeeze(-1)
  return torch.abs(adjusted_target - cmd.robot.data.root_link_pos_w[:, 2])


def root_link_pos_w_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-step base link z position [m]."""
  cmd = env.command_manager.get_term("base_height")
  return cmd.robot.data.root_link_pos_w[:, 2]
