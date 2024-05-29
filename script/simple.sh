#!/usr/bin/env bash
set -x
LOGTXT="/home/fengyaohui/log/simple.out"
nohup python -u main.py >> $LOGTXT 2>&1 &