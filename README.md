# robot

Three manipulation-learning projects, one progression. Each folder is named for
the robot it drives and the simulator it drives it in.

## [panda-libero/](panda-libero) — parallel-jaw manipulation on LIBERO
A Franka Panda in robosuite/LIBERO. `robobench`: a harness for testing
visuomotor architectures with per-episode logging, first-class input ablations
and paired significance testing — plus `robobench.hand`, a vision-based
hand-tracking teleoperation pipeline that turns human video into LIBERO-format
demos.

## [shadow-mujoco/](shadow-mujoco) — dexterous in-hand manipulation
A tendon-driven Shadow Hand in MuJoCo (an Allegro hand is also supported).
In-hand cube reorientation: custom environment, domain randomization, fingertip
tactile, threaded parallel simulation, a PPO teacher on privileged state and a
DAgger student on proprio + tactile.

## [franka-isaac/](franka-isaac) — contact-rich insertion in Isaac Lab
A Franka arm in Isaac Lab. Plug insertion driven by human demonstration: webcam
hand tracking, a teleoperation mapper, differential IK, recorded demos and
behaviour cloning, with residual RL for the last millimetre. The plan is in
[note.md](note.md).

The split is the interesting part: `panda-libero` treats the hand as one scalar
and learns *where to move an arm*; `shadow-mujoco` fixes the arm and learns *how
to exploit contact*. The second problem is the one that does not reduce to the
first. `franka-isaac` is the one that refuses the split — the arm has to travel,
and then the last five millimetres are decided entirely by contact.

Note that `panda-libero` and `franka-isaac` drive the *same* arm, a Franka
Emika Panda; robosuite calls it Panda and Isaac calls it Franka. What separates
them is the simulator and the task, which is why both halves of the name matter.

Each project is self-contained: own venv, own Makefile, own tests
(`make test` in any of the three — no downloads, no GPU, no simulator assets
needed). Running them is where they differ: `panda-libero` and `shadow-mujoco`
run on a laptop, while `franka-isaac` needs an NVIDIA GPU and drives a rented
cloud box.

The Python packages keep their own names — `robobench`, `dexhand`, `insertion`
— since package names cannot contain hyphens and imports should not churn with
directory names.
