from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_lin_vel_yaw(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root linear velocity expressed in the yaw-aligned frame.

  Converts the world-frame linear velocity into a coordinate frame aligned with
  the robot's heading (yaw only), ignoring pitch and roll. Useful when the
  vertical component of motion caused by tilting should not affect the
  horizontal speed evaluation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w),
    asset.data.root_com_lin_vel_w,                       # 应该使用COM处速度而不是link坐标系原点速度
  )


def base_ang_vel_yaw(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root angular velocity expressed in the yaw-aligned frame.

  Transforms the world-frame angular velocity into the yaw-aligned frame,
  effectively removing the pitch and roll components.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w),
    asset.data.root_com_ang_vel_w,                       # 应该使用COM处速度而不是link坐标系原点速度
  )
