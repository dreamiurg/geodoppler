#!/bin/bash
# Runs every variant snapshot; each case takes the shared CPU lock on its own (validation/eval.py).
# Key cases (all correct ones + the two wrong-high ones) go through every variant first: the lock is contended.
cd "$(dirname "$0")/../.." || exit 1
KEY="U1bQZeogcsU J8yvJJS7qTU UsyJgZpSuus GPLUdm_Q9NQ u8qudttB8PI 4CxVxuMVkMc y-xpMDcnQvw y-xpMDcnQvw_akron 0a7boP2vrtY"
for v in heli kflag fbtype narrow; do
  .venv/bin/python validation/eval.py run "$v" --only $KEY
done
for v in base heli kflag fbtype; do
  .venv/bin/python validation/eval.py run "$v"
done
echo queue done
