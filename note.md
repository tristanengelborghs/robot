Build this as a sequence of independently testable components. Do not start with ROS, hand tracking, custom assets and policy training simultaneously.

The first vertical slice should be:

```text
Keyboard → simulated Franka → successful block lift → recorded demonstration
```

Only then replace the keyboard with hand tracking, and only after that train a policy.

## Final architecture

```text
Human demonstration path

Webcam
  ↓
MediaPipe hand tracker
  ↓
ROS 2 teleoperation mapper
  ↓
Differential IK controller
  ↓
Isaac Lab Franka arm
  ↓
Block grasp and lift
  ↓
Demonstration recorder


Autonomous path

Camera + robot state
  ↓
ACT/Diffusion policy
  ↓
Safety filter
  ↓
Differential IK controller
  ↓
Isaac Lab Franka arm
```

## Component 0: Development environment

Use:

- Ubuntu 24.04
- NVIDIA GPU machine
- Isaac Sim and Isaac Lab
- Python
- ROS 2 Jazzy
- C++
- Git
- Docker later

If your main computer is a Mac, run Isaac Sim on a remote Ubuntu GPU machine. You can run hand tracking on the Mac and send ROS messages over the network, but initially keeping everything on one Ubuntu machine will be easier.

### Deliverable

You can:

- Launch Isaac Sim
- Run an Isaac Lab example
- Run a ROS 2 Python publisher and C++ subscriber
- Access a webcam through OpenCV

### Do not continue until

All four work independently.

---

## Component 1: Run an existing insertion task

Start with Isaac Lab’s existing Franka insertion environment:

```text
Isaac-Factory-PegInsert-Direct-v0
```

Isaac Lab currently includes peg insertion, gear meshing and nut fastening as contact-rich manipulation environments. [Use the official environment list as the reference](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments).

At this point, do not change anything. Inspect:

- Robot model
- Plug and socket assets
- Observation space
- Action space
- Control frequency
- Reward function
- Reset procedure
- Success condition
- Contact configuration
- PPO configuration

### Deliverable

Write a short document answering:

- What does one action contain?
- What observations does the policy receive?
- How is insertion success detected?
- Which controller converts actions into joint motion?
- What causes an episode to terminate?
- How are plug and socket poses randomized?

### Done when

You can run, reset and evaluate the environment and explain its main code paths.

---

## Component 2: Validate teleoperation using an existing task

Before teleoperating insertion, validate Isaac Lab’s recording pipeline on a supported manipulation task such as:

```text
Isaac-Stack-Cube-Franka-IK-Rel-v0
```

Use keyboard teleoperation first:

```text
Keyboard
   ↓
Cartesian end-effector delta
   ↓
Differential IK
   ↓
Franka joints
```

Record approximately five demonstrations and replay them. Isaac Lab’s official workflow supports keyboard, SpaceMouse and hand-tracking demonstration recording into HDF5. [Follow the teleoperation and imitation-learning guide](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html).

### Deliverable

- Five recorded episodes
- Successful replay
- A script that prints each dataset’s observation and action shapes
- A visualization of end-effector position over time

### Done when

A replayed episode reproduces the original simulated robot trajectory.

This proves the control and data pipeline before introducing your custom task.

---

## Component 3: Define the block and its randomization

No USD authoring. The stock stacking task already ships a cube of the right
scale, and Component 2 has already recorded and replayed demonstrations with
it. Reuse that asset; spend the saved time on demonstrations and training.

What this component actually produces is the *randomization*, because a policy
trained on one fixed block pose learns a trajectory, not a skill:

- Block position on the table
- Block yaw (the cube is yaw-symmetric every 90 degrees — respect that in any
  pose error, as `stack_sm.YAW_SYMMETRY` already does)
- Block size, within a range the gripper can still close on
- Block mass and friction
- Table friction

Create difficulty configurations rather than one setting:

```text
easy.yaml     fixed pose, nominal physics
medium.yaml   randomized position and yaw
hard.yaml     randomized position, yaw, size, mass, friction
```

### Deliverable

An environment that resets into randomized configurations, with a test that
asserts every randomized field lands inside its band across a few hundred
resets. `shadow-mujoco/randomize.py` already learned this lesson the hard way:
perturb a pristine snapshot each reset, never the live model, or the values
random-walk out of their band and training goes quietly non-stationary.

### Done when

Several hundred resets produce blocks that are always reachable, always
graspable, and never interpenetrating the table.

## Component 4: Build the custom lift environment

Create an Isaac Lab environment containing:

- Franka arm
- Gripper
- Block
- Camera
- Contact or force observations

### Observation space

Initially use privileged simulator state:

```python
observation = {
    "joint_positions": ...,
    "joint_velocities": ...,
    "end_effector_pose": ...,
    "block_pose": ...,
    "gripper_position": ...,
    "contact_force": ...,
}
```

Later replace the block pose with camera input. That swap is the whole point of
Component 12, so keep the two observation views separate from the start rather
than retrofitting the split later.

### Action space

Use relative Cartesian control:

```python
action = {
    "delta_position": [dx, dy, dz],
    "delta_rotation": [droll, dpitch, dyaw],
    "gripper": value,
}
```

This is the same interface the keyboard, the gamepad and eventually the hand
tracker all speak, which is why the teleoperation work already done is not
wasted by the change of task.

### Success condition

Require all of the following:

- Block held in the gripper
- Lifted above a minimum height off the table
- Held for several simulation steps without slipping
- Contact force stays below the safety threshold

The hold requirement matters. Without it a policy that snatches the block and
drops it scores identically to one that grasps it properly, and the failure is
invisible in the aggregate.

### Failure conditions

- Block dropped after grasp
- Block knocked out of reach before grasp
- Excessive contact force
- Robot leaves workspace
- Joint limit reached
- Episode timeout

Log which one fired. A success rate that falls without a failure breakdown
tells you nothing about what to fix.

### Deliverable

The environment resets into randomized starting configurations and correctly
reports success, plus the specific failure mode.

### Done when

The scripted controller from Component 5 completes the easy configuration
consistently, and the failure counters are non-zero on the hard one.

## Component 5: Trim the scripted controller to pick and lift

Mostly already built. `harness/stack_sm.py` is a tested state machine whose
first four states are exactly this task:

```text
HOVER_PICK
      ↓
DESCEND_PICK
      ↓
CLOSE
      ↓
LIFT
```

The remaining states — `HOVER_PLACE`, `DESCEND_PLACE`, `OPEN`, `RETREAT` — are
stacking, not lifting. Cut a lift-only variant rather than deleting them; the
stacking path is what validated the recording pipeline in Component 2 and is
worth keeping runnable as a regression.

Each state keeps its preconditions, target pose, completion condition, timeout
and failure transition. The logic stays pure NumPy and stays tested against the
fake plant on the laptop, because the GPU box bills by the hour and this is the
part that does not need it.

### Why this matters

If the scripted controller cannot complete the task, an imitation or RL policy
probably will not either. It isolates:

- Bad collision geometry
- Incorrect coordinate transforms
- Unstable control
- Incorrect success detection

It is also the demonstration source for everything downstream until hand
tracking arrives, and the baseline every learned policy is measured against.

### Deliverable

- Consistent scripted lift on the easy configuration
- A measured success rate on the hard configuration, which should be lower
- State-transition logs
- Video of a nominal lift and of a representative failure

## Component 6: Create the webcam hand tracker

Use:

- OpenCV for webcam frames
- MediaPipe Hand Landmarker
- ROS 2 Python node

MediaPipe’s live-stream API can asynchronously produce hand landmarks and may drop frames to keep latency low. [Use the current Hand Landmarker API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker).

### Input

```text
Webcam RGB frame
```

### Output

Create a ROS message conceptually containing:

```text
HandState
├── timestamp
├── palm position
├── palm orientation
├── pinch distance
├── tracking confidence
└── hand detected
```

You can derive:

- Palm center: average selected palm landmarks
- Palm direction: vector from wrist toward middle finger
- Palm normal: cross-product of two palm vectors
- Pinch: distance between thumb and index fingertips

### Safety behavior

If:

- No hand is detected
- Confidence is too low
- Timestamp is stale

Then publish:

```text
enabled = false
```

Never keep executing the previous command after losing the hand.

### Deliverable

An OpenCV display showing:

- Hand landmarks
- Palm coordinate axes
- Pinch state
- Confidence
- Current tracking rate

### Done when

The hand coordinates are stable enough while your hand is stationary, and tracking loss safely disables commands.

---

## Component 7: Create the teleoperation mapper

The tracker reports human-hand motion. The mapper converts it into robot motion.

Use relative control:

```python
hand_delta = current_hand_pose - reference_hand_pose
robot_delta = scale * hand_delta
```

Do not map the hand’s absolute room position directly to the robot workspace.

### Add a clutch

Use a keyboard key or gesture:

```text
Clutch pressed:
    hand motion moves robot

Clutch released:
    robot remains still
    human can reposition hand
```

### Mapping

| Human input | Robot action |
|---|---|
| Palm left/right | End-effector X |
| Palm up/down | End-effector Z |
| Palm toward/away from camera | End-effector Y |
| Wrist rotation | End-effector rotation |
| Pinch | Close gripper |
| Open pinch | Open gripper |
| Clutch released | Hold position |

### Filtering

Add:

- Low-pass filtering
- Dead zone
- Position scaling
- Rotation scaling
- Maximum velocity
- Maximum acceleration
- Workspace clipping

### ROS output

Publish continuous commands as a topic:

```text
/teleop/cartesian_delta
```

Use `geometry_msgs/TwistStamped` or a custom message containing:

```text
linear delta
angular delta
gripper command
enabled flag
```

Topics are appropriate for continuous command streams; ROS actions are better for cancellable, longer-running tasks. [ROS documents those interface distinctions here](https://docs.ros.org/en/jazzy/How-To-Guides/Topics-Services-Actions.html).

### Deliverable

Use your hand to move a marker in RViz or a simple simulated cube before connecting it to the robot.

### Done when

You can accurately move the marker to several targets without abrupt jumps.

---

## Component 8: Connect teleoperation to the Franka

Connect the mapper to Isaac Lab’s differential-IK controller.

```text
Teleop delta pose
        ↓
Target end-effector pose
        ↓
Differential IK
        ↓
Target joint positions
        ↓
Franka
```

Add a safety layer before IK:

```python
safe_command = safety_filter(raw_command)
```

The safety filter checks:

- Workspace boundaries
- Maximum position delta
- Maximum rotation delta
- Stale command
- Tracking status
- Joint limits
- Contact-force threshold

### Deliverable

Complete the easy lift task using your hand, with keyboard fine adjustment.

### Done when

You can produce at least five successful lifts without simulation instability.

---

## Component 9: Implement demonstration recording

Record the simulated robot state—not only the human hand state.

At every policy timestep save:

```text
Observation
├── camera RGB
├── optional depth
├── joint positions
├── joint velocities
├── end-effector pose
├── gripper state
├── force/contact data
└── task instruction

Action
├── Cartesian position delta
├── Cartesian rotation delta
└── gripper command

Metadata
├── timestamp
├── episode ID
├── randomization parameters
├── success
└── failure reason
```

Also save raw hand commands separately for debugging, but do not use them as the primary policy target.

### Recording controls

Create ROS services:

```text
/start_episode
/stop_episode
/reset_episode
/discard_episode
```

Only successful demonstrations go into the primary training set. Keep failures in a separate dataset for later analysis.

### Deliverable

Collect:

- 10 successful easy demonstrations
- 10 unsuccessful attempts
- Replay all successful episodes

### Done when

Every recorded episode can be replayed and its success label is correct.

---

## Component 10: Build a dataset validator

Before training, create automated validation.

Check:

- Equal observation/action sequence lengths
- Monotonically increasing timestamps
- No NaN or infinite values
- Actions remain within limits
- Images decode correctly
- No long missing-data gaps
- Success state agrees with lift height and hold duration
- Correct train/validation episode split

Generate a dataset report:

```text
Number of episodes
Success/failure count
Episode duration distribution
Action distribution
Workspace coverage
Block-pose distribution
Grasp-point distribution on the block
Maximum contact force
Camera frame rate
```

### Done when

One command validates the complete dataset and produces plots.

---

## Component 11: Train a state-based behavior-cloning baseline

Do not start with images.

Input:

```text
joint state
end-effector pose
block pose
gripper state
```

Output:

```text
Cartesian delta
rotation delta
gripper command
```

Use a small PyTorch MLP.

This verifies:

- Dataset loading
- Normalization
- Training
- Checkpointing
- Inference
- Closed-loop rollout

### Deliverable

Compare:

- Scripted controller
- Random policy
- Behavior-cloning MLP

### Done when

The learned policy performs better than random in closed-loop evaluation.

---

## Component 12: Train the visual policy

Replace the privileged block pose with:

- RGB image
- Robot proprioception
- Optional depth

Train ACT using LeRobot.

Progress incrementally:

```text
Stage 1: fixed camera and lighting
Stage 2: randomized object pose
Stage 3: randomized lighting
Stage 4: camera perturbations
Stage 5: unseen block sizes and colours
```

Compare dataset sizes:

```text
25 demos
50 demos
100 demos
200 demos
```

### Deliverable

A plot of task success versus demonstration count.

### Done when

The policy lifts the block closed-loop from images and robot state.

---

## Component 13: Add residual reinforcement learning

Freeze the imitation policy.

Train PPO to output a small correction:

```python
final_action = imitation_action + residual_scale * rl_action
```

Give RL low-dimensional observations:

- Grasp-pose error
- Contact force per finger
- Recent actions
- End-effector velocity
- Block height above the table

Keep residual corrections small.

Compare:

- PPO from scratch
- Imitation only
- Imitation plus residual PPO

The residual policy should focus on the approach and grasp — the last few centimetres, where behaviour cloning is weakest and small pose errors decide whether the fingers close on the block or beside it. It should not be relearning the whole reach.

---

## Component 14: Split everything into ROS 2 nodes

Only do this after the Isaac-only pipeline works.

```text
hand_tracking_node
        ↓
teleop_mapper_node
        ↓
safety_controller_node
        ↓
isaac_bridge_node

policy_inference_node
        ↓
safety_controller_node

evaluation_node
        ↓
isaac_bridge_node

recording_node subscribes to all relevant topics
```

Use:

- `rclpy` for hand tracking, inference, recording and evaluation
- `rclcpp` for the safety controller
- `tf2` for coordinate frames
- Topics for continuous observations and commands
- Services for episode reset/recording
- Actions for complete task execution
- rosbag2 for debugging
- RViz 2 for transform and pose visualization

---

## Component 15: Evaluation and portfolio

Create a fixed evaluation suite:

- Nominal conditions
- Random block poses
- Lighting changes
- Camera calibration errors
- Control latency
- Sensor noise
- Friction and mass changes
- Unseen block sizes
- Disturbance during the lift

Report:

- Task-success rate
- Grasp-success rate
- Hold-success rate (grasped, lifted, still held)
- Recovery rate after a slip
- Peak contact force
- Completion time
- Inference latency

The final demonstration should show:

1. Hand-tracked teleoperation.
2. A recorded demonstration replay.
3. ACT performing autonomously.
4. A failed grasp.
5. Residual RL or recovery improving the attempt.
6. ROS 2 nodes and RViz.
7. Quantitative evaluation results.

## Strict implementation order

Follow this exact dependency order:

```text
Isaac installation
→ existing insertion task (studied, not built on)
→ existing keyboard teleoperation
→ custom assets
→ custom environment
→ scripted controller
→ webcam tracker
→ teleoperation mapper
→ IK integration
→ recording
→ dataset validation
→ state behavior cloning
→ visual ACT policy
→ residual RL
→ ROS 2 decomposition
→ MuJoCo transfer
```

The key rule is: every component must have a standalone test before connecting it to the next one. This prevents you from debugging camera tracking, coordinate transforms, robot control and policy learning at the same time.
---

# Later project: the dexterous hand

**Status, 2026-08-30: this became `shadow-isaac`, but with a different task.**
The hand and the reasoning below survived; the task did not. Rather than
unscrewing a bottle in MuJoCo, the project catches a falling ball on a palm-up
Shadow Hand in Isaac Lab, because catching needs the massive parallelism Isaac
buys -- a random policy essentially never catches, and rare success is what
thousands of environments are for. The scripted catcher is at 87.9%.

Bottle opening remains a good later task and the design work below still holds,
in particular the finding that the Shadow Hand has no forearm roll, the
hinge-plus-slide `polycoef` screw, and the argument for live teleoperation over
record-then-replay. What Isaac taught that MuJoCo would not have: the stock
asset stands the hand vertical, and nothing tells you -- the metric read 100%
caught while the hand lay on its side.

---

Block lifting proves the pipeline. It does not, on its own, prove dexterity —
a parallel-jaw gripper is one scalar, open or closed, and everything
interesting happens in the arm. This is the project that inverts that, and it
is the one meant to carry the portfolio. It lives in `shadow-mujoco`, not
here: same hand, same simulator, and it reuses that repo's environment, PPO,
DAgger, domain randomization, threaded vec env, tactile extraction and
per-episode evaluation rather than starting over.

## The task: opening a bottle

Unscrewing a screw cap, which requires **finger gaiting** — grip, rotate,
release, re-grip, rotate again. A cap needs two to three full turns and no
single grip yields more than about 90–120 degrees, so the regrasp is the task
rather than a refinement of it. This is the property a parallel-jaw gripper
cannot reproduce at any level of skill.

It is also a real-world task an interviewer recognises instantly, and it is
far less crowded than in-hand cube reorientation (OpenAI 2018, DeXtreme 2022,
and a hundred repositories since).

## Findings that already constrain the design

**The Shadow Hand cannot roll.** The Menagerie model exposes only `rh_WRJ1`
and `rh_WRJ2` — wrist flexion/extension and radial/ulnar deviation. There is
no forearm pronation/supination, because on the real robot that comes from the
arm. So the choice is forced:

- Zero roll — every degree of cap rotation comes from fingertips walking
  around the cap. The purest dexterity result available, brutally hard, and
  mismatched to human demonstrations because a human demonstrator *will* roll
  their wrist and there is no actuator to receive it.
- **One forearm roll joint at the attachment frame.** Take this. Human wrist
  roll maps cleanly and the natural strategy emerges on its own: twist to the
  roll limit, release, counter-rotate the wrist, re-grip, twist again. It is a
  small change to `scene.py`, which already attaches the hand under a frame.
  A full arm becomes a later upgrade rather than a dependency.

The arm contributes one degree of freedom — the roll. Everything else is
fingers. The roll-on/off ablation then measures exactly how much of the task
the arm is doing.

**The screw is cheap to simulate.** MuJoCo has no screw joint, but a cap body
with a hinge about z and a slide along z, coupled by an `<equality><joint>`
with a linear `polycoef`, *is* a screw. When the slide clears the thread
length the environment sets `eq_active = 0` and the cap comes free in the
hand. No thread geometry, no solver instability, no USD authoring.

**Demonstrations must be live teleoperation, not record-then-replay.** Miming
the motion in free air and replaying it offline yields almost nothing: contact
timing will not line up and virtually no episode actually unscrews anything.
The human has to watch the simulator and close the loop, adapting when a
finger slips.

**Camera placement is load-bearing, and this task is the good case.**
`robobench.hand.pose.palm_frame` gates on Kabsch residual because roll about
the palm normal is its least reliable component — and unscrewing is *entirely*
roll about the palm normal. A top-down camera over the bottle keeps the palm
broadside to the lens through the whole twist, making the rotation that
matters an in-image-plane rotation, which is the best-conditioned case rather
than the worst. The real risk is the regrasp, where fingers occlude each other
and MediaPipe begins hallucinating fingertips. That is what the confidence
gate is for.

**Retargeting is new code.** `robobench.hand.retarget` targets a 7-D OSC
action; this needs 20 actuators plus the roll. What transfers is the lessons
in its docstring, each of which was a silent failure once: deltas rather than
poses, rotations transformed by conjugation, resampling onto the control clock
before use, Slerp rather than componentwise interpolation, and a deadband that
carries its residual.

## Setup

```text
Bottle welded to the table. Single-handed; bimanual stabilisation is out of
scope and is stated as such rather than quietly ignored.

Success  cap unscrewed through the full thread travel AND still held
Failure  cap dropped, cross-threaded, excessive contact force, timeout

easy     1 turn,  low thread friction
medium   2 turns, moderate friction
hard     3 turns, high stiction, randomised cap radius, pitch, friction,
         bottle pose
```

## The comparison, which is the actual result

Three policies, one evaluation harness, paired initial states, and the
permutation test from `robobench.stats`:

- **PPO from scratch** should plateau near the single-grip limit. Releasing
  the cap to regrasp reads as catastrophic reward loss to an exploring policy —
  a textbook exploration trap, and a legible one.
- **Imitation from human demonstrations** should break that ceiling on a small
  fraction of the compute, because human demonstrations contain the regrasp
  natively; the demonstrator does it without thinking about it.
- **Imitation plus residual RL** should beat both, cleaning up the slip
  recovery that behaviour cloning is worst at.

"RL plateaus at the single-grip limit, human demonstrations break through it,
and residual RL adds slip recovery" is a finding rather than a table. Report
compute alongside success rate; the compute ratio is half the point.

Ablations that fall out for free: tactile on/off (it should matter most at the
regrasp, where slip happens), demonstration-count curve, and roll DoF on/off.

## Risk

The retargeting spike is the schedule risk — 21 MediaPipe landmarks onto 20
tendon-coupled actuators is not a solved mapping and can eat a week. Timebox
it, and keep a scripted gaiting controller as the fallback demonstration
source so the comparison ships either way.
