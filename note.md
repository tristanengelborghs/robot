Build this as a sequence of independently testable components. Do not start with ROS, hand tracking, custom assets and policy training simultaneously.

The first vertical slice should be:

```text
Keyboard → simulated Franka → successful insertion → recorded demonstration
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
Plug insertion
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

## Component 3: Create the keyed plug and socket

Create simple geometry first:

- Rectangular plug
- Matching socket
- One asymmetrical feature, such as a cut corner
- Chamfered socket entrance
- Separate visual and collision geometry

Use USD for Isaac Sim assets.

Start with generous tolerances:

```text
Plug width:          W
Socket interior:     W + generous clearance
Socket entrance:     funnel or chamfer
```

Create difficulty configurations rather than one extremely precise asset:

```text
easy.yaml
medium.yaml
hard.yaml
```

Randomize:

- Plug position
- Plug rotation
- Socket position
- Clearance
- Friction

### Deliverable

A scene where you can manually move the plug:

- Through the socket when correctly aligned
- Against the socket when incorrectly aligned

### Done when

The physics behaves as expected and incorrect orientations are mechanically rejected.

Do not start training until collision geometry is stable.

---

## Component 4: Build the custom insertion environment

Create an Isaac Lab environment containing:

- Franka arm
- Gripper
- Plug
- Socket
- Camera
- Contact or force observations

### Observation space

Initially use privileged simulator state:

```python
observation = {
    "joint_positions": ...,
    "joint_velocities": ...,
    "end_effector_pose": ...,
    "plug_pose": ...,
    "socket_pose": ...,
    "gripper_position": ...,
    "contact_force": ...,
}
```

Later replace plug and socket poses with camera input.

### Action space

Use relative Cartesian control:

```python
action = {
    "delta_position": [dx, dy, dz],
    "delta_rotation": [droll, dpitch, dyaw],
    "gripper": value,
}
```

### Success condition

Require all of the following:

- Correct plug orientation
- Correct lateral alignment
- Minimum insertion depth
- Plug remains inside for several simulation steps
- Contact force stays below the safety threshold

### Failure conditions

- Object dropped
- Excessive contact force
- Robot leaves workspace
- Joint limit reached
- Episode timeout

### Deliverable

The environment can reset into randomized starting configurations and correctly report success or failure.

### Done when

A simple scripted controller completes the fixed-pose, easy version consistently.

---

## Component 5: Implement a scripted controller

Before human demonstrations, create a controller using ground-truth poses.

Break insertion into states:

```text
APPROACH_PLUG
      ↓
GRASP
      ↓
LIFT
      ↓
MOVE_ABOVE_SOCKET
      ↓
ALIGN
      ↓
INSERT
      ↓
VERIFY
```

Each state has:

- Preconditions
- Target pose
- Completion condition
- Timeout
- Failure transition

Example:

```python
if state == ALIGN:
    target_position = socket_position + pre_insert_offset
    target_rotation = socket_orientation

    if pose_error < threshold:
        state = INSERT
```

### Why this matters

If the scripted controller cannot complete the task, an imitation or RL policy probably will not either. It helps isolate:

- Bad collision geometry
- Incorrect coordinate transforms
- Unstable control
- Incorrect success detection

### Deliverable

- Successful scripted insertion
- State-transition logs
- Video of nominal completion
- Video of rejected incorrect alignment

---

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

Complete the easy insertion task using your hand and keyboard fine adjustment.

### Done when

You can produce at least five successful insertions without simulation instability.

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
- Success state agrees with insertion depth
- Correct train/validation episode split

Generate a dataset report:

```text
Number of episodes
Success/failure count
Episode duration distribution
Action distribution
Workspace coverage
Plug-pose distribution
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
plug pose
socket pose
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

Replace privileged plug/socket poses with:

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
Stage 5: unseen plug/socket configurations
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

The policy performs closed-loop insertion from images and robot state.

---

## Component 13: Add residual reinforcement learning

Freeze the imitation policy.

Train PPO to output a small correction:

```python
final_action = imitation_action + residual_scale * rl_action
```

Give RL low-dimensional observations:

- Alignment error
- Contact force
- Recent actions
- End-effector velocity
- Insertion depth

Keep residual corrections small.

Compare:

- PPO from scratch
- Imitation only
- Imitation plus residual PPO

The residual policy should focus on final alignment and insertion, not the complete pick-and-place trajectory.

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
- Random object poses
- Random socket poses
- Lighting changes
- Camera calibration errors
- Control latency
- Sensor noise
- Friction changes
- Unseen plug geometry
- Disturbance during insertion

Report:

- Task-success rate
- Grasp-success rate
- Alignment-success rate
- Insertion-success rate
- Recovery rate
- Peak contact force
- Completion time
- Inference latency

The final demonstration should show:

1. Hand-tracked teleoperation.
2. A recorded demonstration replay.
3. ACT performing autonomously.
4. A failed insertion.
5. Residual RL or recovery improving the attempt.
6. ROS 2 nodes and RViz.
7. Quantitative evaluation results.

## Strict implementation order

Follow this exact dependency order:

```text
Isaac installation
→ existing insertion task
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