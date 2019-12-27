#!/bin/bash

echo "Starting koda server..."

export FLASK_APP=/src/server.py
export FLASK_RUN_PORT=8888
export KODA_MODEL_PATH=/src/models/unet-70.h5

flask run --host=0.0.0.0 --without-threads
