"""RM-style velocity command generator.

    lin_vel/ang_vel at COM origin) is made explicit via mjlab's named
    properties (``root_link_*`` / ``root_com_*``). See #TODO markers.
    ``heading_error = -heading_w`` is preserved as-is (no ``wrap_to_pi``).See #NOTE markers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

  from .commands_cfg import RMVelocityCommandCfg


class RMVelocityCommand(CommandTerm):
  r"""Command generator that generates a velocity command in SE(2) from RM-style distribution.

  The command comprises of a linear velocity in x and y direction and an angular velocity around
  the z-axis. It is given in the robot's base frame.
  """

  cfg: "RMVelocityCommandCfg"
  """The configuration of the command generator."""

  def __init__(self, cfg: "RMVelocityCommandCfg", env: ManagerBasedRlEnv):
    """Initialize the command generator.

    Args:
      cfg: The configuration of the command generator.
      env: The environment.
    """
    # initialize the base class
    super().__init__(cfg, env)

    # obtain the robot entity
    self.robot: Entity = env.scene[cfg.entity_name]

    # make sure configurations are reasonable
    assert 0.0 <= self.cfg.rel_standing_envs <= 1.0
    assert 0.0 <= self.cfg.rel_pure_rotation_envs <= 1.0
    assert 0.0 <= self.cfg.rel_heading_envs <= 1.0
    assert self.cfg.rel_pure_rotation_envs + self.cfg.rel_standing_envs + self.cfg.rel_heading_envs <= 1.0

    # create buffers to store the command
    # -- command: x vel, y vel, yaw vel, heading
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_pure_rotation_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.pure_rotation_vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  def __str__(self) -> str:
    """Return a string representation of the command generator."""
    msg = "RMVelocityCommand:\n"
    msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
    msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
    msg += f"\tPure rotation probability: {self.cfg.rel_pure_rotation_envs}\n"
    msg += f"\tStanding probability: {self.cfg.rel_standing_envs}\n"
    msg += f"\tHeading probability: {self.cfg.rel_heading_envs}"
    return msg

  """
  Properties
  """

  @property
  def command(self) -> torch.Tensor:
    """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
    return self.vel_command_b

  """
  Implementation specific functions.
  """

  def _update_metrics(self):
    # Metrics are computed in metrics.py (registered via env_cfgs) so that
    # MetricsManager handles per-step normalization correctly. Keep this as
    # a no-op since CommandTerm requires the method to exist.
    pass

  def _resample_command(self, env_ids: torch.Tensor):
    # sample velocity commands
    r = torch.empty(len(env_ids), device=self.device)
    # -- linear velocity - x direction
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    # -- linear velocity - y direction
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    # -- ang vel yaw - rotation around z
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    # update standing and pure rotation envs
    uniform_sample = r.uniform_(0.0, 1.0)
    self.is_standing_env[env_ids] = uniform_sample <= self.cfg.rel_standing_envs
    self.is_pure_rotation_env[env_ids] = uniform_sample >= 1.0 - self.cfg.rel_pure_rotation_envs
    self.is_heading_env[env_ids] = (
      (uniform_sample > self.cfg.rel_standing_envs)
      & (uniform_sample <= self.cfg.rel_standing_envs + self.cfg.rel_heading_envs)
    )
    # sample pure rotation velocity
    self.pure_rotation_vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pure_rotation_ang_vel_z)

  def _update_command(self):
    """Post-processes the velocity command."""
    # Enforce standing (i.e., zero velocity command) for standing envs
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    pure_rotation_env_ids = self.is_pure_rotation_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[pure_rotation_env_ids, :] = self.pure_rotation_vel_command_b[pure_rotation_env_ids, :]
    heading_env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
    # NOTE: IsaacLab reference uses heading_error = -heading_w (target heading = 0, no wrap_to_pi).
    heading_error = -self.robot.data.heading_w[heading_env_ids]
    self.vel_command_b[heading_env_ids, 2] = torch.clip(
      self.cfg.heading_control_stiffness * heading_error,
      min=self.cfg.ranges.ang_vel_z[0],
      max=self.cfg.ranges.ang_vel_z[1],
    )

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    ranges = self.cfg.ranges
    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}", initial_value=max_val, step=0.1, min=0.0, max=10.0
        )
        slider = server.gui.add_slider(
          label, min=-max_val, max=max_val, step=0.05, initial_value=0.0
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)
      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # Visualization.
  # NOTE: IsaacLab used USD VisualizationMarkers (GREEN_ARROW_X / BLUE_ARROW_X) with
  # prim_path and post-construction scale edits. mjlab has no equivalent: it injects
  # a DebugVisualizer into _debug_vis_impl and draws via add_arrow(start, end, color).
  # The (0.5,0.5,0.5) arrow scale is emulated by scaling the start->end vector.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw goal velocity (green) and current velocity (blue) arrows above the robot."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    # TODO(IsaacLab->mjlab): IsaacLab root_pos_w is at link origin.
    # mjlab makes this explicit: root_link_pos_w.
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    # TODO(IsaacLab->mjlab): IsaacLab root_quat_w is at link origin.
    # mjlab makes this explicit: root_link_quat_w.
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    # TODO(IsaacLab->mjlab): IsaacLab root_lin_vel_w is at COM origin.
    # mjlab makes this explicit: root_com_lin_vel_w.
    lin_vel_ws = self.robot.data.root_com_lin_vel_w.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_w = lin_vel_ws[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Goal velocity arrow (green): command linear xy in body frame.
      goal_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      goal_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(goal_from, goal_to, color=(0.2, 0.6, 0.2, 0.7), width=0.015)

      # Current velocity arrow (blue): actual linear velocity in world frame.
      cur_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cur_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_w[0], lin_vel_w[1], lin_vel_w[2]])) * scale
      )
      visualizer.add_arrow(cur_from, cur_to, color=(0.2, 0.2, 0.6, 0.7), width=0.015)
