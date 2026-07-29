from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def links_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize z-axis body linear velocity using L2 squared kernel.
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel_z = asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, 2]     # 应该使用COM处速度而不是link坐标系原点速度
  return torch.sum(torch.square(lin_vel_z), dim=1)


def links_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize xy-axis body angular velocity using L2 squared kernel.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_xy = asset.data.body_com_ang_vel_w[:, asset_cfg.body_ids, :2]   # 应该使用COM处速度而不是link坐标系原点速度
  return torch.sum(torch.square(ang_vel_xy), dim=(1, 2))


def track_base_height_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Penalize asset height deviation from its target using L2 squared kernel.

  Note:
    For flat terrain, target height is in the world frame. For rough terrain,
    sensor readings (RayCastSensor) can adjust the target height to account
    for the terrain.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target_height = env.command_manager.get_command(command_name).squeeze(-1)
  if sensor_cfg is not None:
    sensor = env.scene[sensor_cfg.name]
    # mjlab RayCastData exposes `hit_pos_w` (world-frame hit positions).
    adjusted_target_height = target_height + torch.mean(
      sensor.data.hit_pos_w[..., 2], dim=1
    )
  else:
    adjusted_target_height = target_height
  return torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)


def track_base_height_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Reward asset height tracking using exponential kernel.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target_height = env.command_manager.get_command(command_name).squeeze(-1)
  if sensor_cfg is not None:
    sensor = env.scene[sensor_cfg.name]
    adjusted_target_height = target_height + torch.mean(
      sensor.data.hit_pos_w[..., 2], dim=1
    )
  else:
    adjusted_target_height = target_height
  return torch.exp(
    -torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target_height)
    / std**2
  )


def track_lin_vel_xy_yaw_frame_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
  robot frame (yaw frame) using an exponential kernel.

  The world-frame linear velocity is rotated into the yaw-only frame (roll and
  pitch removed) so that tracking is evaluated relative to the robot's heading
  direction, not its tilted body frame.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel_yaw = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w),
    asset.data.root_com_lin_vel_w,                                                # 应该使用COM处速度而不是link坐标系原点速度
  )
  lin_vel_error = torch.sum(
    torch.square(
      env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
    ),
    dim=1,
  )
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking of angular velocity commands (yaw) in world frame using
  exponential kernel.

  Uses the world-frame angular velocity's z component directly. Since the z
  component of angular velocity is invariant under yaw rotation, this is
  equivalent to comparing against the body-frame yaw rate, but matches the
  IsaacLab reference exactly.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_error = torch.square(
    env.command_manager.get_command(command_name)[:, 2]
    - asset.data.root_com_ang_vel_w[:, 2]                                        # 应该使用COM处速度而不是link坐标系原点速度
  )
  return torch.exp(-ang_vel_error / std**2)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  threshold: float,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize undesired contacts as the number of violations above a threshold.

  Ported from IsaacLab ``mdp.undesired_contacts``.

  Requires a ``ContactSensor`` configured with ``reduce="netforce"`` and
  ``history_length > 0`` (set to the decimation value so the buffer covers one
  policy step). Uses ``data.force_history`` and takes the max over the history
  dimension, matching the IsaacLab reference which checks whether the force
  exceeded the threshold at *any* substep within the last policy step.

  The set of monitored bodies is determined by the ContactSensor's ``primary``
  configuration, not by a SceneEntityCfg filter.

  Args:
    threshold: Force magnitude (N) above which a contact is considered undesired.
    sensor_name: Name of the ContactSensor in the scene.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  force_history = contact_sensor.data.force_history  # [B, N, H, 3]
  assert force_history is not None, (
    "undesired_contacts requires ContactSensorCfg.history_length > 0 and "
    "'force' in fields, with reduce='netforce'."
  )
  # norm over force dim → [B, N, H], then max over history → [B, N]
  force_norm = torch.norm(force_history, dim=-1)  # [B, N, H]
  max_force = torch.max(force_norm, dim=-1)[0]  # [B, N]
  is_contact = max_force > threshold  # [B, N]
  return torch.sum(is_contact, dim=1)  # [B]
