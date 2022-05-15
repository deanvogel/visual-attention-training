#!/usr/bin/python
envs=(
    Breakout-v0
    MsPacman-v4
)
t = 2e6
for i in "${envs[@]}"; do
    python training.py "$i" $t
done
sleep 3