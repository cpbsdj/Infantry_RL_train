import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg,JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactSensorCfg,
  ContactMatch,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from infantry.tasks.velocity.mdp import observations as infantry_observations
from infantry.tasks.velocity.mdp import rewards as infantry_rewards
from infantry.tasks.velocity.mdp.commands_cfg import HeightCommandCfg, RMVelocityCommandCfg
from infantry.tasks.velocity.mdp.metrics import command_error_height, command_error_vel_xy, command_error_vel_yaw, root_link_pos_w_z
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_infantry_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  
  ##
  # Sensors
  ##

  # Contact sensor for undesired_contacts reward.
  contact_forces = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(
      mode="body",
      pattern="base_link",
      entity="robot",
    ),
    secondary=None,  # Any contact counts.
    fields=("force",),
    reduce="netforce",
    history_length=3,
  )

  # Height scan sensor for the base height command.
  height_scan = RayCastSensorCfg(
    name="height_scan",
    frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=infantry_observations.base_lin_vel_yaw,
      # noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=5,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=infantry_observations.base_ang_vel_yaw,
      # noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=5,
      flatten_history_dim=True,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      # noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=5,
      flatten_history_dim=True,
    ),
    "velocity_commands": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    ),
    "height_commands": ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "base_height"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=(".*hip_joint", ".*knee_joint"),
            ),
        },
      # noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=5,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=(".*hip_joint", ".*knee_joint",".*wheel_joint"),
            ),
        },
      # noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=5,
      flatten_history_dim=True,
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=5,
      flatten_history_dim=True,
    ),
  }

  critic_terms = {
    **actor_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    "error_vel_xy": MetricsTermCfg(func=command_error_vel_xy),
    "error_vel_yaw": MetricsTermCfg(func=command_error_vel_yaw),
    "error_height": MetricsTermCfg(func=command_error_height),
    "root_link_pos_w_z": MetricsTermCfg(func=root_link_pos_w_z),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*hip_joint", ".*knee_joint"),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    ),
    "joint_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=(".*wheel_joint",),
        scale=1.0,  # Override per-robot.
        use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
   
    # "base_velocity": UniformVelocityCommandCfg(
    #   entity_name="robot",
    #   resampling_time_range=(10.0, 10.0),
    #   rel_standing_envs=0.1,
    #   rel_heading_envs=0.4,
    #   rel_forward_envs=0.2,
    #   heading_command=True,
    #   heading_control_stiffness=0.5,
    #   debug_vis=True,
    #   ranges=UniformVelocityCommandCfg.Ranges(
    #     lin_vel_x=(-1.5, 1.5),
    #     lin_vel_y=(-0.5, 0.5),
    #     ang_vel_z=(-1.0, 1.0),
    #     heading=(-math.pi, math.pi),
    #   ),
    # ),
   
    "base_velocity": RMVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.1,
      rel_pure_rotation_envs=0.1,
      rel_heading_envs=0.4,
      heading_control_stiffness=1.0,
      debug_vis=True,
      ranges=RMVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-1.0, 1.0),
        pure_rotation_ang_vel_z=(-8.0, 8.0),
      ),
    ),
    "base_height": HeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      sensor_name="height_scan",
      ranges=HeightCommandCfg.Ranges(
        height_z=(0.15, 0.40),
      ),
      debug_vis=True,
    ),
  }

  ##
  # Events
  ##

  events = {
    # reset
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (-0.0, 0.0),
          "roll": (-0.0, 0.0),
          "pitch": (-0.0, 0.0),
          "yaw": (-0.0, 0.0),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),


    # startup
    "physics_material": EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*wheel_l_collision")),
            "ranges": (0.4, 1.0),
            "operation": "abs",
        },
    ),
    "scale_all_link_masses": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "ranges": (0.9, 1.1),
            "operation": "scale",
        },
    ),
    "add_base_mass": EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "ranges": (-1.0, 1.0),
            "operation": "add",
        },
    ),
    "base_external_force_torque": EventTermCfg(
        mode="reset",
        func=mdp.apply_external_force_torque,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-10.0, 10.0),
        },
    ),


    # interval
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(10.0, 15.0),
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
        },
      },
    ),
    
  }

  ##
  # Rewards
  ##

  rewards = {
    # ── Regularization for smoothness ──
    "lin_vel_z_l2": RewardTermCfg(
      func=infantry_rewards.links_lin_vel_z_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=infantry_rewards.links_ang_vel_xy_l2,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    ),
    "dof_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.0e-6,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),

    # ── Penalties for undesired behaviors ──
    "termination_penalty": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-200.0,
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*knee_joint"])},
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-20.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"])},
    ),
    "undesired_contacts": RewardTermCfg(
      func=infantry_rewards.undesired_contacts,
      weight=-10.0,
      params={
        "sensor_name": "contact_forces",
        "threshold": 1.0,
      },
    ),

    # ── Rewards for desired behaviors ──
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=infantry_rewards.track_lin_vel_xy_yaw_frame_exp,
      weight=8.0,
      params={
        "command_name": "base_velocity",
        "std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
      },
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=infantry_rewards.track_ang_vel_z_world_exp,
      weight=2.0,
      params={
        "command_name": "base_velocity",
        "std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
      },
    ),
    "base_height_l2": RewardTermCfg(
      func=infantry_rewards.track_base_height_l2,
      weight=-200.0,                         # -100
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        "sensor_cfg": SceneEntityCfg("height_scan"),
        "command_name": "base_height",
      },
    ),
    "base_height_exp": RewardTermCfg(
      func=infantry_rewards.track_base_height_exp,
      weight=4.0,                            # 2.0
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        "sensor_cfg": SceneEntityCfg("height_scan"),
        "command_name": "base_height",
        "std": 0.02,                          # 0.05
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": 1.0},
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "base_velocity"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "base_velocity",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
      ),
      sensors=(contact_forces, height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    # events=events,
    rewards=rewards,
    terminations=terminations,
    # curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
