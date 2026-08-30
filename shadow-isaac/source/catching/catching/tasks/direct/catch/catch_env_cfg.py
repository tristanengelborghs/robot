"""Scene and task configuration for the catch environment.

Every number here that could have been a guess has been measured on the box
instead, because each one of them was wrong at least once and none of them
failed loudly:

* the body-name patterns and DOF count, from `make bodies`
* the palm's facing, from `make orient` -- the stock asset stands the hand
  vertical with the fingers up, and a ball dropped on that rolls off the tips
* the fingertip height above the palm, reported by the env at every reset

The pattern worth keeping: a contact-sensor regex that matches nothing does not
raise, it reports no contacts; an asset whose palm faces sideways still runs
happily. Both surface much later as "the policy cannot learn". Re-measure after
any Isaac Lab upgrade -- asset names and poses move between releases.
"""

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class CatchEnvCfg(DirectRLEnvCfg):
    # -- curriculum stage: "drop" | "toss" | "pop" ------------------------
    # Catching is learned before throwing: the drop stage has no ballistic
    # phase and therefore no credit-assignment gap.
    stage: str = "drop"

    # -- rates ------------------------------------------------------------
    # 240 Hz physics, decimation 1 => 240 Hz control. The first smoke run used
    # decimation 2 and a 5 cm drop: 12 control steps of fall, which the
    # validator would have refused had anything called it. It does now, from
    # CatchEnv.__init__.
    decimation: int = 1
    episode_length_s: float = 4.0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 240.0, render_interval=decimation)

    # -- spaces -----------------------------------------------------------
    # joint_pos + joint_vel + ball_rel_pos(3) + ball_vel(3)
    #   + intercept_xy(2) + intercept_valid(1) + contact_forces(16) + action(20)
    # The env asserts the assembled width against this on the first step, so a
    # hand with a different DOF count is a loud error rather than a silent one.
    num_hand_dofs: int = 24          # verified: 24 joints, 20 of them actuated
    # 1 palm + 5 proximal + 5 middle + 5 distal. Measured, not chosen: with the
    # sensor on fingertips alone, 83% of the env-steps where a ball was
    # demonstrably resting on the hand reported ZERO contacts. The ball is
    # carried by the palm and the phalanges below the tips, so a sensor that
    # watches only tips cannot see a catch happen.
    num_contact_bodies: int = 16
    action_space: int = 20
    observation_space: int = 24 + 24 + 3 + 3 + 2 + 1 + 16 + 20
    state_space: int = 0

    # -- hand -------------------------------------------------------------
    # activate_contact_sensors is not optional: without it the fingertip bodies
    # carry no PhysX contact reporter and the ContactSensor refuses to
    # initialise ("could not find any bodies with contact reporter API"). It is
    # off by default in the stock asset, so it has to be turned on here.
    #
    # PALM UP -- and found by SEARCH, not by algebra. Two attempts at composing
    # a corrective quaternion put the hand somewhere the arithmetic did not
    # predict, so `make palmup` writes all 24 axis-aligned orientations into a
    # running simulator and reads back where the palm actually points. This is
    # the winner: palm normal [0 0 1], fingers horizontal, 0.4 deg off.
    #
    # The measurement that matters is taken with the CUPPED reset pose and the
    # palm's sign anchored on the finger curl. An earlier attempt measured the
    # default straight-fingered pose and signed it by assuming the thumb sits
    # on the palmar side; it was wrong, and the resulting hand lay on its side
    # with the palm facing left and the thumb up -- while the scripted catcher
    # reported 100% caught, because a ball wedged in a sideways crook satisfies
    # every is_secured condition without being a catch.
    # Written at RESET, not at spawn. `init_state.rot` is composed with the
    # asset's own USD transform, so the same quaternion that gives palm-up
    # through `write_root_pose_to_sim` leaves the hand 90 deg off through
    # init_state -- which is exactly how the searched value still produced a
    # hand on its side. `make palmup` validates the runtime path, so the env
    # uses the runtime path.
    hand_rot: tuple = (0.707107, 0.0, 0.0, -0.707107)
    hand_pos: tuple = (0.0, 0.0, 0.5)
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=SHADOW_HAND_CFG.spawn.replace(activate_contact_sensors=True),
    )
    # Refuse to run if the palm is not actually up, within this many degrees.
    max_palm_tilt_deg: float = 20.0
    palm_body_expr: str = ".*palm"
    contact_body_expr: str = ".*(palm|proximal|middle|distal)"
    action_scale: float = 1.0

    # -- ball -------------------------------------------------------------
    # Restitution is deliberately near zero. A lively ball bounces straight
    # back out of a good catch, and the policy is then being punished for
    # succeeding.
    ball_radius: float = 0.03
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
                # A ball at ~1 m/s crosses 4 mm per physics step; without a
                # tightened contact offset the solver can let it settle
                # visibly inside a fingertip.
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1)),
            # Enabled on the ball too, so contact filtering (see contact_cfg)
            # remains available without another round trip to the box.
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
    )

    # Two sensors, because they answer two different questions and only one of
    # them can be filtered.
    #
    # This one is the tactile OBSERVATION: which parts of the hand are loaded.
    # It cannot be filtered to the ball -- PhysX wants one filter entry per
    # (body, env) pair, so a 16-body sensor across 64 envs demands 1024 filter
    # prims and finds 64 ("expected 1024, found 64"). Filtering in Isaac Lab
    # needs a sensor matching a single prim per environment. So it stays
    # unfiltered and includes finger-on-finger contact, which a real tactile
    # skin would report too.
    contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*(palm|proximal|middle|distal)",
        track_air_time=False,
        history_length=0,
    )

    # This one decides whether the ball is HELD, and it sidesteps the filtering
    # limit entirely: the ball is one body per environment, and the only thing
    # it can touch is the hand -- once it reaches the ground the episode has
    # already ended as `dropped`. So its unfiltered net contact force IS the
    # hand-ball force, with none of the finger-on-finger contamination that
    # tripped `excess_force` on every step of every episode.
    ball_contact_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Ball",
        track_air_time=False,
        history_length=0,
    )
    contact_threshold: float = 0.1   # newtons; above this the ball is touched

    # -- task shape (mirrors harness.catch.CatchConfig) -------------------
    # Zero, unlike the in-hand reorientation sibling. There, settling let the
    # cube come to rest on the palm before scoring began. Here the ball is in
    # free fall from the first step and every step of that fall IS the task --
    # 20 settle steps was longer than the entire 5 cm drop, so the policy was
    # blind for the whole episode and every episode ended `dropped`.
    settle_steps: int = 0
    # Clearance above the FINGERTIPS, not above the palm -- the ball used to
    # spawn inside the fingers. The real fall to palm level is
    # spawn_height + hand_reach, and that is what the validator uses. This is
    # the curriculum variable; `Curriculum` widens it on measured success.
    spawn_height: float = 0.01
    # Fingertips above the palm origin. Measured 5.9 cm with the palm up and
    # the reset pose cupped; kept at 7.2 as a deliberate OVER-estimate, because
    # the error is asymmetric -- too high only lengthens the drop, too low
    # spawns the ball inside the fingers and PhysX flings it out at 253 N. It
    # was 16.9 cm while the hand stood vertical: that was the length of the
    # fingers, not the depth of a basket. The env prints the measured value at
    # the first reset.
    hand_reach: float = 0.072
    max_spawn_height: float = 0.30
    spawn_jitter: float = 0.01      # lateral, metres

    # Where along the hand the ball is dropped, as a fraction of the distance
    # from the palm body ORIGIN to the knuckle row. Zero drops it on the palm
    # origin, which sits back toward the wrist -- the ball landed on the wrist
    # rather than in the cup. The knuckles are ~9.5 cm out, so 0.5 puts it
    # ~4.7 cm forward, near the middle of the basket.
    #
    # Expressed as a fraction of a measured vector rather than as a world
    # offset so that it survives a change to hand_rot: "toward the fingers" is
    # derived from the hand, not assumed about the world.
    #
    # Swept with the scripted catcher (64 envs, 2000 steps, lead 0.15):
    #     0.00  0.0 cm   97.4% caught    <- lands on the wrist
    #     0.15  1.3 cm   95.2%           <- still read as too far back
    #     0.25  2.1 cm   87.9%           <- default: chosen on the viewer
    #     0.35  2.9 cm   79.8%
    #     0.50  4.2 cm   81.3%
    # The fall-off is the real difficulty gradient of the task -- the further
    # toward the fingertips the ball arrives, the less hand there is under it.
    # Worth randomising later; it is a more honest difficulty knob than drop
    # height alone, because it varies WHERE the catch has to happen.
    spawn_toward_knuckles: float = 0.25

    catch_radius: float = 0.08
    secure_speed: float = 0.25
    # One, over the widened body set. Two fingertips was never a definition of
    # "held" -- it was a definition of "pinched". A ball cradled in a palm is
    # caught. The kinematic conditions carry most of the weight anyway: a ball
    # in free fall gains 2.45 m/s over the 0.25 s hold window, so staying under
    # secure_speed for hold_steps is not possible without support.
    min_contacts: int = 1
    hold_steps: int = 60          # 0.25 s at 240 Hz

    drop_below: float = 0.10
    drop_lateral: float = 0.25
    max_force: float = 40.0
    launch_deadline: int = 480    # 2 s at 240 Hz (TOSS/POP only)

    track_scale: float = 2.0
    secure_bonus: float = 50.0
    hold_scale: float = 1.0
    apex_scale: float = 10.0
    drop_penalty: float = 20.0
    force_penalty: float = 0.5
    action_rate_penalty: float = 0.01
    ctrl_penalty: float = 0.001

    # -- scene ------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.75, replicate_physics=True)
