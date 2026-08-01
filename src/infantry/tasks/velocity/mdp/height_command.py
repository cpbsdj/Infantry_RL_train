"""Height command generator.

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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm

if TYPE_CHECKING:
  import viser

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

    # Metrics are registered in env_cfgs (metrics.py) via MetricsManager for
    # correct per-step normalization; no command-side metric accumulation.

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_slider: viser.GuiSliderHandle | None = None
    self._joystick_get_env_idx: Callable[[], int] | None = None

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
    # Metrics are computed in metrics.py (registered via env_cfgs) so that
    # MetricsManager handles per-step normalization correctly. Keep this as
    # a no-op since CommandTerm requires the method to exist.
    pass

  def _resample_command(self, env_ids: torch.Tensor):
    # sample height commands
    r = torch.empty(len(env_ids), device=self.device)
    self.height_command_z[env_ids, 0] = r.uniform_(*self.cfg.ranges.height_z)

  def _update_command(self):
    """Post-processes the height command."""
    pass

  # GUI.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create a height slider in the Viser viewer."""
    lo, hi = self.cfg.ranges.height_z
    default = (lo + hi) / 2.0

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "height_z",
        min=float(lo),
        max=float(hi),
        step=0.01,
        initial_value=float(default),
      )

    self._joystick_enabled = enabled
    self._joystick_slider = slider
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      assert self._joystick_slider is not None
      idx = self._joystick_get_env_idx()
      self.height_command_z[idx, 0] = self._joystick_slider.value

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # Draw target height marker: a green sphere at (robot_x, robot_y, target_z).
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue
      target_z = float(self.height_command_z[batch, 0])
      # Green sphere at target height, below the robot's xy position
      target_pos = np.array([base_pos_w[0], base_pos_w[1], target_z])
      visualizer.add_sphere(target_pos, radius=0.03, color=(0.0, 0.8, 0.0, 0.9))
