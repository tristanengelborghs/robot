# Isaac-Factory-PegInsert-Direct-v0

Dumped by `scripts/inspect_env.py`. Do not edit by hand.

## task

`Isaac-Factory-PegInsert-Direct-v0`

## observation_space

`Box(-inf, inf, (1, 19), float32)`

## action_space

`Box(-inf, inf, (1, 6), float32)`

## num_observations

`19`

## num_actions

`6`

## sim_dt

`0.008333333333333333`

## decimation

`8`

## control_hz

`15.0`

## physics_hz

`120.0`

## episode_length_s

`10.0`

## robot_cfg

```json
{
  "class_type": "isaaclab.assets.articulation.articulation:Articulation",
  "prim_path": "/World/envs/env_.*/Robot",
  "spawn": {
    "func": "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd",
    "visible": true,
    "semantic_tags": null,
    "copy_from_source": true,
    "spawn_path": null,
    "mass_props": null,
    "deformable_props": null,
    "rigid_props": {
      "rigid_body_enabled": null,
      "kinematic_enabled": null,
      "disable_gravity": true,
      "linear_damping": 0.0,
      "angular_damping": 0.0,
      "max_linear_velocity": 1000.0,
      "max_angular_velocity": 3666.0,
      "max_depenetration_velocity": 5.0,
      "max_contact_impulse": 1e+32,
      "enable_gyroscopic_forces": true,
      "retain_accelerations": null,
      "solver_position_iteration_count": 192,
      "solver_velocity_iteration_count": 1,
      "sleep_threshold": null,
      "stabilization_threshold": null
    },
    "collision_props": {
      "collision_enabled": null,
      "contact_offset": 0.005,
      "rest_offset": 0.0,
      "torsional_patch_radius": null,
      "min_torsional_patch_radius": null
    },
    "activate_contact_sensors": true,
    "scale": null,
    "articulation_props": {
      "articulation_enabled": null,
      "fix_root_link": null,
      "enabled_self_collisions": false,
      "solver_position_iteration_count": 192,
      "solver_velocity_iteration_count": 1,
      "sleep_threshold": null,
      "stabilization_threshold": null
    },
    "fixed_tendons_props": null,
    "spatial_tendons_props": null,
    "joint_drive_props": null,
    "visual_material_path": "material",
    "visual_material": null,
    "physics_material_path": "material",
    "physics_material": null,
    "usd_path": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Factory/franka_mimic.usd",
    "variants": null
  },
  "init_state": {
    "pos": [
      0.0,
      0.0,
      0.0
    ],
    "rot": [
      0.0,
      0.0,
      0.0,
      1.0
    ],
    "lin_vel": [
      0.0,
      0.0,
      0.0
    ],
    "ang_vel": [
      0.0,
      0.0,
      0.0
    ],
    "joint_pos": {
      "panda_joint1": 0.00871,
      "panda_joint2": -0.10368,
      "panda_joint3": -0.00794,
      "panda_joint4": -1.49139,
      "panda_joint5": -0.00083,
      "panda_joint6": 1.38774,
      "panda_joint7": 0.0,
      "panda_finger_joint2": 0.04
    },
    "joint_vel": {
      ".*": 0.0
    }
  },
  "collision_group": 0,
  "debug_vis": false,
  "disable_shape_checks": null,
  "articulation_root_prim_path": null,
  "soft_joint_pos_limit_factor": 1.0,
  "actuators": {
    "panda_arm1": {
      "class_type": "isaaclab.actuators.actuator_pd:ImplicitActuator",
      "joint_names_expr": [
        "panda_joint[1-4]"
      ],
      "effort_limit": null,
      "velocity_limit": null,
      "effort_limit_sim": 87,
      "velocity_limit_sim": 124.6,
      "stiffness": 0.0,
      "damping": 0.0,
      "armature": 0.0,
      "friction": 0.0,
      "dynamic_friction": null,
      "viscous_friction": null
    },
    "panda_arm2": {
      "class_type": "isaaclab.actuators.actuator_pd:ImplicitActuator",
      "joint_names_expr": [
        "panda_joint[5-7]"
      ],
      "effort_limit": null,
      "velocity_limit": null,
      "effort_limit_sim": 12,
      "velocity_limit_sim": 149.5,
      "stiffness": 0.0,
      "damping": 0.0,
      "armature": 0.0,
      "friction": 0.0,
      "dynamic_friction": null,
      "viscous_friction": null
    },
    "panda_hand": {
      "class_type": "isaaclab.actuators.actuator_pd:ImplicitActuator",
      "joint_names_expr": [
        "panda_finger_joint[1-2]"
      ],
      "effort_limit": null,
      "velocity_limit": null,
      "effort_limit_sim": 40.0,
      "velocity_limit_sim": 0.04,
      "stiffness": 7500.0,
      "damping": 173.0,
      "armature": 0.0,
      "friction": 0.1,
      "dynamic_friction": null,
      "viscous_friction": null
    }
  },
  "actuator_value_resolution_debug_print": false
}
```

## task_cfg

```json
{
  "robot_cfg": {
    "robot_usd": "",
    "franka_fingerpad_length": 0.017608,
    "friction": 0.75
  },
  "name": "peg_insert",
  "fixed_asset_cfg": {
    "usd_path": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Factory/factory_hole_8mm.usd",
    "diameter": 0.0081,
    "height": 0.025,
    "base_height": 0.0,
    "friction": 0.75,
    "mass": 0.05
  },
  "held_asset_cfg": {
    "usd_path": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Factory/factory_peg_8mm.usd",
    "diameter": 0.007986,
    "height": 0.05,
    "friction": 0.75,
    "mass": 0.019
  },
  "asset_size": 8.0,
  "hand_init_pos": [
    0.0,
    0.0,
    0.047
  ],
  "hand_init_pos_noise": [
    0.02,
    0.02,
    0.01
  ],
  "hand_init_orn": [
    3.1416,
    0.0,
    0.0
  ],
  "hand_init_orn_noise": [
    0.0,
    0.0,
    0.785
  ],
  "unidirectional_rot": false,
  "fixed_asset_init_pos_noise": [
    0.05,
    0.05,
    0.05
  ],
  "fixed_asset_init_orn_deg": 0.0,
  "fixed_asset_init_orn_range_deg": 360.0,
  "held_asset_pos_noise": [
    0.003,
    0.0,
    0.003
  ],
  "held_asset_rot_init": 0.0,
  "ee_success_yaw": 0.0,
  "action_penalty_ee_scale": 0.0,
  "action_grad_penalty_scale": 0.0,
  "num_keypoints": 4,
  "keypoint_scale": 0.15,
  "keypoint_coef_baseline": [
    5,
    4
  ],
  "keypoint_coef_coarse": [
    50,
    2
  ],
  "keypoint_coef_fine": [
    100,
    0
  ],
  "success_threshold": 0.04,
  "engage_threshold": 0.9,
  "duration_s": 10.0,
  "fixed_asset": {
    "class_type": "isaaclab.assets.articulation.articulation:Articulation",
    "prim_path": "/World/envs/env_.*/FixedAsset",
    "spawn": {
      "func": "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd",
      "visible": true,
      "semantic_tags": null,
      "copy_from_source": true,
      "spawn_path": null,
      "mass_props": {
        "mass": 0.05,
        "density": null
      },
      "deformable_props": null,
      "rigid_props": {
        "rigid_body_enabled": null,
        "kinematic_enabled": null,
        "disable_gravity": false,
        "linear_damping": 0.0,
        "angular_damping": 0.0,
        "max_linear_velocity": 1000.0,
        "max_angular_velocity": 3666.0,
        "max_depenetration_velocity": 5.0,
        "max_contact_impulse": 1e+32,
        "enable_gyroscopic_forces": true,
        "retain_accelerations": null,
        "solver_position_iteration_count": 192,
        "solver_velocity_iteration_count": 1,
        "sleep_threshold": null,
        "stabilization_threshold": null
      },
      "collision_props": {
        "collision_enabled": null,
        "contact_offset": 0.005,
        "rest_offset": 0.0,
        "torsional_patch_radius": null,
        "min_torsional_patch_radius": null
      },
      "activate_contact_sensors": true,
      "scale": null,
      "articulation_props": null,
      "fixed_tendons_props": null,
      "spatial_tendons_props": null,
      "joint_drive_props": null,
      "visual_material_path": "material",
      "visual_material": null,
      "physics_material_path": "material",
      "physics_material": null,
      "usd_path": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Factory/factory_hole_8mm.usd",
      "variants": null
    },
    "init_state": {
      "pos": [
        0.6,
        0.0,
        0.05
      ],
      "rot": [
        0.0,
        0.0,
        0.0,
        1.0
      ],
      "lin_vel": [
        0.0,
        0.0,
        0.0
      ],
      "ang_vel": [
        0.0,
        0.0,
        0.0
      ],
      "joint_pos": {},
      "joint_vel": {}
    },
    "collision_group": 0,
    "debug_vis": false,
    "disable_shape_checks": null,
    "articulation_root_prim_path": null,
    "soft_joint_pos_limit_factor": 1.0,
    "actuators": {},
    "actuator_value_resolution_debug_print": false
  },
  "held_asset": {
    "class_type": "isaaclab.assets.articulation.articulation:Articulation",
    "prim_path": "/World/envs/env_.*/HeldAsset",
    "spawn": {
      "func": "isaaclab.sim.spawners.from_files.from_files:spawn_from_usd",
      "visible": true,
      "semantic_tags": null,
      "copy_from_source": true,
      "spawn_path": null,
      "mass_props": {
        "mass": 0.019,
        "density": null
      },
      "deformable_props": null,
      "rigid_props": {
        "rigid_body_enabled": null,
        "kinematic_enabled": null,
        "disable_gravity": true,
        "linear_damping": 0.0,
        "angular_damping": 0.0,
        "max_linear_velocity": 1000.0,
        "max_angular_velocity": 3666.0,
        "max_depenetration_velocity": 5.0,
        "max_contact_impulse": 1e+32,
        "enable_gyroscopic_forces": true,
        "retain_accelerations": null,
        "solver_position_iteration_count": 192,
        "solver_velocity_iteration_count": 1,
        "sleep_threshold": null,
        "stabilization_threshold": null
      },
      "collision_props": {
        "collision_enabled": null,
        "contact_offset": 0.005,
        "rest_offset": 0.0,
        "torsional_patch_radius": null,
        "min_torsional_patch_radius": null
      },
      "activate_contact_sensors": true,
      "scale": null,
      "articulation_props": null,
      "fixed_tendons_props": null,
      "spatial_tendons_props": null,
      "joint_drive_props": null,
      "visual_material_path": "material",
      "visual_material": null,
      "physics_material_path": "material",
      "physics_material": null,
      "usd_path": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/Factory/factory_peg_8mm.usd",
      "variants": null
    },
    "init_state": {
      "pos": [
        0.0,
        0.4,
        0.1
      ],
      "rot": [
        0.0,
        0.0,
        0.0,
        1.0
      ],
      "lin_vel": [
        0.0,
        0.0,
        0.0
      ],
      "ang_vel": [
        0.0,
        0.0,
        0.0
      ],
      "joint_pos": {},
      "joint_vel": {}
    },
    "collision_group": 0,
    "debug_vis": false,
    "disable_shape_checks": null,
    "articulation_root_prim_path": null,
    "soft_joint_pos_limit_factor": 1.0,
    "actuators": {},
    "actuator_value_resolution_debug_print": false
  }
}
```

## scene_cfg

```json
{
  "num_envs": 1,
  "env_spacing": 2.0,
  "lazy_sensor_update": true,
  "replicate_physics": true,
  "filter_collisions": true,
  "clone_in_fabric": true
}
```

## sim_cfg

```json
{
  "device": "cuda:0",
  "dt": 0.008333333333333333,
  "gravity": [
    0.0,
    0.0,
    -9.81
  ],
  "physics_prim_path": "/physicsScene",
  "physics_material": {
    "func": "isaaclab.sim.spawners.materials.physics_materials:spawn_rigid_body_material",
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "compliant_contact_stiffness": null,
    "compliant_contact_damping": null,
    "friction_combine_mode": null,
    "restitution_combine_mode": null
  },
  "use_fabric": true,
  "render_interval": 1,
  "enable_scene_query_support": false,
  "use_newton_actuators": false,
  "physics": {
    "class_type": "isaaclab_physx.physics.physx_manager:PhysxManager",
    "solver_type": 1,
    "solve_articulation_contact_last": false,
    "min_position_iteration_count": 1,
    "max_position_iteration_count": 192,
    "min_velocity_iteration_count": 0,
    "max_velocity_iteration_count": 1,
    "enable_scene_query_support": false,
    "enable_ccd": false,
    "enable_stabilization": false,
    "enable_external_forces_every_iteration": false,
    "enable_enhanced_determinism": false,
    "bounce_threshold_velocity": 0.2,
    "friction_offset_threshold": 0.01,
    "friction_correlation_distance": 0.00625,
    "gpu_max_rigid_contact_count": 8388608,
    "gpu_max_rigid_patch_count": 8388608,
    "gpu_found_lost_pairs_capacity": 2097152,
    "gpu_found_lost_aggregate_pairs_capacity": 33554432,
    "gpu_total_aggregate_pairs_capacity": 2097152,
    "gpu_collision_stack_size": 268435456,
    "gpu_heap_capacity": 67108864,
    "gpu_temp_buffer_capacity": 16777216,
    "gpu_max_num_partitions": 1,
    "gpu_max_soft_body_contacts": 1048576,
    "gpu_max_particle_contacts": 1048576
  },
  "render": {
    "enable_translucency": null,
    "enable_reflections": null,
    "enable_global_illumination": null,
    "antialiasing_mode": null,
    "enable_dlssg": null,
    "enable_dl_denoiser": null,
    "dlss_mode": null,
    "enable_direct_lighting": null,
    "samples_per_pixel": null,
    "enable_shadows": null,
    "enable_ambient_occlusion": null,
    "dome_light_upper_lower_strategy": null,
    "max_bounces": null,
    "split_glass": null,
    "split_clearcoat": null,
    "split_rough_reflection": null,
    "ambient_light_intensity": null,
    "ambient_occlusion_denoiser_mode": null,
    "view_tile_limit": null,
    "carb_settings": null,
    "rendering_mode": null
  },
  "create_stage_in_memory": false,
  "logging_level": "WARNING",
  "save_logs_to_file": true,
  "log_dir": null,
  "visualizer_cfgs": []
}
```

## robot_joint_names

```json
[
  "panda_joint1",
  "panda_joint2",
  "panda_joint3",
  "panda_joint4",
  "panda_joint5",
  "panda_joint6",
  "panda_joint7",
  "panda_finger_joint1",
  "panda_finger_joint2"
]
```

## robot_body_names

```json
[
  "panda_link0",
  "panda_link1",
  "panda_link2",
  "panda_link3",
  "panda_link4",
  "panda_link5",
  "panda_link6",
  "panda_link7",
  "force_sensor",
  "panda_hand",
  "panda_leftfinger",
  "panda_rightfinger",
  "panda_fingertip_centered"
]
```

## scene_keys

```json
[
  "terrain",
  "robot",
  "fixed_asset",
  "held_asset"
]
```
