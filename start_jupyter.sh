#!/bin/bash

echo "Starting docker container for Jupyter notebook..."

export KODA_MODEL_PATH=/src/models/unet-70.h5

docker run --rm -p 8888:8888 -v "$(pwd):/src" -it koda