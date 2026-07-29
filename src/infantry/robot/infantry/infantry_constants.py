"""Infantry constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

INFANTRY_XML: Path = (
    Path(__file__).parent / "xmls" / "infantry.xml"
)
assert INFANTRY_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(INFANTRY_XML))


##
# Actuator config.
##

INFANTRY_LEG_ACTUATOR_CFG = BuiltinPdActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_knee_joint"),
  effort_limit=70,
  stiffness=60.0,
  damping=6.0,
  armature=0.01,
)
INFANTRY_WHEEL_ACTUATOR_CFG = BuiltinPdActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
  effort_limit=17.0,
  stiffness=0.0,
  damping=3.0,
  armature=0.01,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.3),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_knee_joint": 0.0,
    ".*_wheel_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# ##
# # Collision config.
# ##

# _foot_regex = "^[FR][LR]_foot_collision$"

# # This disables all collisions except the feet.
# # Furthermore, feet self collisions are disabled.
# FEET_ONLY_COLLISION = CollisionCfg(
#   geom_names_expr=(_foot_regex,),
#   contype=0,
#   conaffinity=1,
#   condim=3,
#   priority=1,
#   friction=(0.6,),
#   solimp=(0.9, 0.95, 0.023),
# )

# # This enables all collisions.
# # Foot collisions are given custom condim, friction.
# FULL_COLLISION = CollisionCfg(
#   geom_names_expr=(".*_collision",),
#   # Harden all collision geoms.
#   solref=(0.01, 1),
#   # Configure feet colliders. Other colliders are frictionless (condim=1).
#   condim={_foot_regex: 6, ".*_collision": 1},
#   priority={_foot_regex: 1},
#   friction={_foot_regex: (1, 5e-3, 5e-4)},
# )

##
# Final config.
##

INFANTRY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    INFANTRY_LEG_ACTUATOR_CFG,
    INFANTRY_WHEEL_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_infantry_robot_cfg() -> EntityCfg:
  """Get a fresh Infantry robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    # collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=INFANTRY_ARTICULATION,
    init_state=INIT_STATE,
  )


# INFANTRY_ACTION_SCALE: dict[str, float] = {}
# for a in INFANTRY_ARTICULATION.actuators:
#   assert isinstance(a, BuiltinPositionActuatorCfg)
#   e = a.effort_limit
#   s = a.stiffness
#   names = a.target_names_expr
#   assert e is not None
#   for n in names:
#     INFANTRY_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_infantry_robot_cfg())

  viewer.launch(robot.spec.compile())
