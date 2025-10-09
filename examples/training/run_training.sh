#!/bin/sh
ps  -elf | grep 'ray' | awk '{print $4}' | xargs kill -9 
Xvfb :4 -maxclients 1024 &
export DISPLAY=:4
RAY_PDB=1 RAY_memory_monitor_refresh_ms=0 ray start --head --resources='{"database": 10000, "rollout_workers": 1, "wandb": 1}' --port 9899
python /alex/examples/training/test_train.py