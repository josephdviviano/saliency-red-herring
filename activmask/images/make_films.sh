#!/bin/bash

# It's funny how the colors of the real world only seem really real when you
# viddy them on the screen.

for folder in $(ls -d */); do

    if [ ! -f ${folder}/train_valid.mp4 ]; then
        ffmpeg -y -i ${folder}/image-train-%03d.png \
            -vcodec libx264 ${folder}/tmp_train.mp4
        ffmpeg -y -i ${folder}/image-valid-%03d.png \
            -vcodec libx264 ${folder}/tmp_valid.mp4
        ffmpeg \
            -i ${folder}/tmp_train.mp4 \
            -i ${folder}/tmp_valid.mp4 \
            -filter_complex vstack=inputs=2 \
            -vcodec libx264 \
            ${folder}/train_valid.mp4
        rm ${folder}/tmp_*
    fi
done

