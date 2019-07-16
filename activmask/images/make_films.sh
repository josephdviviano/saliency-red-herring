#!/bin/bash

# It's funny how the colors of the real world only seem really real when you
# viddy them on the screen.

for folder in $(ls -d */); do
    ffmpeg -y -i ${folder}/image-train-%03d.png -vcodec libx264 ${folder}/train.mp4
    ffmpeg -y -i ${folder}/image-valid-%03d.png -vcodec libx264 ${folder}/valid.mp4
done
