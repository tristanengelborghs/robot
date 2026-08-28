# robot

Two manipulation-learning projects, one progression:

## [1dof/](1dof) — parallel-jaw manipulation on LIBERO
`robobench`: a harness for testing visuomotor architectures on LIBERO with
per-episode logging, first-class input ablations and paired significance
testing — plus `robobench.hand`, a vision-based hand-tracking teleoperation
pipeline that turns human video into LIBERO-format demos.

## [dexhand/](dexhand) — dexterous in-hand manipulation
In-hand cube reorientation on a tendon-driven Shadow Hand in MuJoCo:
custom environment, domain randomization, fingertip tactile, threaded
parallel simulation, a PPO teacher on privileged state and a DAgger student
on proprio + tactile.

The split is the interesting part: `1dof` treats the hand as one scalar and
learns *where to move an arm*; `dexhand` fixes the arm and learns *how to
exploit contact*. The second problem is the one that does not reduce to the
first.

Each project is self-contained: own venv, own Makefile, own tests
(`make test` in either — no downloads, no GPU, no simulator assets needed).
