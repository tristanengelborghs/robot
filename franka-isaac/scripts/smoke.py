"""Run the headless peg-insert task on the GPU host and stream its output.

This is the check that the rented box is still capable of what it was rented
for: Isaac Lab starts, the stock Factory task loads its Franka and its plug,
and physics steps on the GPU. It changes nothing and trains nothing.

The run is capped, because ``random_agent.py`` loops forever, and killed
explicitly afterwards, because a capped ``docker exec`` kills only the local
client while the simulator inside the container keeps running -- and billing.
"""

import subprocess
import sys

from insertion import remote

TIMEOUT_S = 420


def main() -> int:
    cmd = remote.smoke()
    print(f"$ {cmd}\n", flush=True)
    try:
        proc = subprocess.run(cmd, shell=True, timeout=TIMEOUT_S)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        # Expected: the random agent has no natural end.
        code = 0
        print(f"\n[smoke] stopped after {TIMEOUT_S}s, as intended", flush=True)

    print("[smoke] cleaning up any simulator left in the container", flush=True)
    subprocess.run(remote.kill_sim(), shell=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
