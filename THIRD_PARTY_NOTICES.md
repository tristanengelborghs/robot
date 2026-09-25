# Third-party notices

Most of this repository is original work. The exceptions are listed here.

## NVIDIA Isaac Lab (BSD-3-Clause)

`franka-isaac/` contains source files copied or generated from
[Isaac Lab](https://github.com/isaac-sim/IsaacLab). They keep their original
copyright headers:

- `franka-isaac/reference/isaaclab_source/` — upstream files captured as reference
  (keyboard devices, dataset handlers, the lift and stack task configurations,
  including their robomimic agent configs)
- `franka-isaac/reference/factory_source/` — the stock Factory insertion task, captured
  as reference
- `franka-isaac/scripts/rl_games/`, `franka-isaac/scripts/list_envs.py`,
  `franka-isaac/scripts/random_agent.py`, `franka-isaac/scripts/zero_agent.py`
- `franka-isaac/source/insertion/` — the extension scaffold produced by Isaac Lab's
  project template

Where these files were modified, the changes are this repository's.

Isaac Lab's license text, as published upstream:

```
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).

All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Fetched at install time, not redistributed

LIBERO, robosuite, MuJoCo Menagerie (Shadow Hand, Allegro Hand) and rl_games
are installed or fetched into gitignored directories by the setup scripts and
are covered by their own licenses.
