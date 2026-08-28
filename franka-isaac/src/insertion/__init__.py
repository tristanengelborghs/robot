"""Contact-rich plug insertion on a Franka arm, in Isaac Lab.

The two sibling projects split the manipulation problem in half. ``panda-libero``
collapses the hand to one scalar and learns *where to move an arm*; ``shadow-mujoco``
bolts the arm down and learns *how to exploit contact*. Insertion is the case
that refuses the split: the arm has to travel several centimetres, and then the
last five millimetres are decided entirely by contact, with the socket occluded
by the plug the moment it matters.

Simulator note: this is Isaac Lab, not MuJoCo, and it is the one project here
that does not run on the M1 (see ``shadow-mujoco`` for why the other two do). Isaac
Sim needs an NVIDIA RTX GPU, so development happens on a rented cloud box --
``remote.py`` is the seam that keeps that fact from leaking into the rest of
the code. Everything in this package is either pure Python that runs and tests
anywhere, or a command handed to ``remote`` for execution on the GPU host.
"""
