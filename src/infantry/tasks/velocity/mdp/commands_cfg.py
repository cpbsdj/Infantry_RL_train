"""Configuration for RM velocity and height command generators.

Faithful port of the IsaacLab ``commands_cfg_reference`` to mjlab.
Key differences from the IsaacLab original:
  - ``@configclass`` (IsaacLab) -> ``@dataclass(kw_only=True)`` (mjlab).
  - ``asset_name`` -> ``entity_name`` (mjlab naming convention).
  - ``MISSING`` sentinel for required fields -> no default (dataclass makes
    them required).
  - IsaacLab's ``VisualizationMarkersCfg`` + ``BLUE/GREEN_ARROW_X_MARKER_CFG``
    + ``FRAME_MARKER_CFG`` have no mjlab equivalent. Visualization is driven
    by a ``DebugVisualizer`` injected into ``_debug_vis_impl``; per-term
    visual params live in a ``VizCfg`` dataclass (scale, z_offset) mirroring
    mjlab's ``UniformVelocityCommandCfg``.
  - ``HeightCommandCfg.sensor_cfg: SceneEntityCfg`` -> ``sensor_name: str``
    (mjlab's SceneEntityCfg is for entity components, not sensors).
  - mjlab's ``CommandTermCfg`` requires a ``build(env)`` method (IsaacLab used
    ``class_type``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.managers.command_manager import CommandTermCfg

from .height_command import HeightCommand
from .velocity_command import RMVelocityCommand


@dataclass(kw_only=True)
class RMVelocityCommandCfg(CommandTermCfg):
  """Configuration for the velocity command generator specified for RM scenarios."""

  entity_name: str
  """Name of the entity in the environment for which the commands are generated."""

  rel_standing_envs: float = 0.0
  """The sampled probability of environments that should be standing still. Defaults to 0.0."""

  rel_pure_rotation_envs: float = 0.0
  """The sampled probability of environments that should have pure rotation. Defaults to 0.0."""

  rel_heading_envs: float = 0.0
  """The sampled probability of environments where the robots follow the heading-based
  angular velocity command. Defaults to 0.0."""

  heading_control_stiffness: float = 1.0
  """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

  @dataclass
  class Ranges:
    """Uniform distribution ranges for the velocity commands."""

    lin_vel_x: tuple[float, float]
    """Range for the linear-x velocity command (in m/s)."""

    lin_vel_y: tuple[float, float]
    """Range for the linear-y velocity command (in m/s)."""

    ang_vel_z: tuple[float, float]
    """Range for the angular-z velocity command (in rad/s)."""

    pure_rotation_ang_vel_z: tuple[float, float]
    """Range for the angular-z velocity command (in rad/s) for pure rotation environments."""

  ranges: Ranges
  """Distribution ranges for the velocity commands."""

  @dataclass
  class VizCfg:
    """Visualization settings for debug rendering (mjlab DebugVisualizer)."""

    z_offset: float = 0.5
    """Height offset above the robot base where arrows are drawn."""

    scale: float = 0.5
    """Scale factor applied to velocity vectors when drawing arrows.
    Mirrors the IsaacLab (0.5, 0.5, 0.5) arrow marker scale."""

  viz: VizCfg = field(default_factory=VizCfg)
  """Visualization settings."""

  def build(self, env) -> RMVelocityCommand:
    return RMVelocityCommand(self, env)


@dataclass(kw_only=True)
class HeightCommandCfg(CommandTermCfg):
  """Configuration for the height command generator."""

  entity_name: str
  """Name of the entity in the environment for which the commands are generated."""

  sensor_name: str = ""
  """Name of the RayCastSensor in the scene used to adjust the target height on
  rough terrain. Empty string disables sensor-based adjustment (flat terrain)."""

  @dataclass
  class Ranges:
    """Uniform distribution ranges for the height commands."""

    height_z: tuple[float, float]
    """Range for the base height command (in m)."""

  ranges: Ranges
  """Distribution ranges for the height commands."""

  def build(self, env) -> HeightCommand:
    return HeightCommand(self, env)
