# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-rich plug insertion on a Franka arm in Isaac Lab.

See ``../../../README.md``; the plan is in ``../../../../note.md``.
"""

# Register Gym environments.
from .tasks import *

# NOTE: The UI extension (``ui_extension_example.py``) imports ``omni.ext``, which only exists
# while Kit is running. Kit loads it via the ``...ui_extension_example`` ``[[python.module]]``
# entry in ``config/extension.toml``; it is intentionally not imported here so that importing
# this package stays omni-free for headless use (e.g. Gym registration before SimulationApp).
