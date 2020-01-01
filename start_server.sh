#!/bin/bash

echo "Starting docker container with Flask Server..."

docker run --rm -p 8888:8888 -v "$(pwd):/src" -it koda /src/web/start_flask.sh