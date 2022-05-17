#!/usr/bin/python
envs=(
    Breakout-v0
    MsPacman-v4
)
for i in "${envs[@]}"; do
    python training.py "$i" 200000
done
sleep 3