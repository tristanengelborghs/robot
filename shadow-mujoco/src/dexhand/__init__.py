"""In-hand cube reorientation on dexterous hands.

The sibling project (../panda-libero) collapses the hand to one scalar: a parallel-jaw
gripper is open or closed, and everything interesting happens in the arm. This
project is the opposite bet — the arm barely moves and everything interesting
happens in contact: 20 actuators, tendon-coupled underactuated distal joints,
and a cube that is only controllable through intermittent frictional contact
with five fingertips. That is the problem class ("multi-contact manipulation on
underactuated, tactile-rich hardware"), and no amount of gripper work prepares
a policy for it.

Simulator note: this is MuJoCo, not Isaac Lab, for one honest reason — Isaac
requires an NVIDIA RTX GPU and this repo's development machine is an M1 Mac.
MuJoCo is the stronger contact/tendon simulator anyway; what Isaac buys is
scale (thousands of parallel envs), and the seams here are cut so a port is a
rewrite of `vec.py`, not of the task: the env logic never assumes who steps
the physics.
"""

__version__ = "0.1.0"
