#!/bin/bash

salloc -t 3:0:0 -A rpp-bengioy --gres=gpu:1 --mem=32G
