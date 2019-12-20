```
sudo docker build -t koda .
```

Per avviare jupyter

```
sudo docker run --rm -p 8888:8888 -it koda
```

Poi navigare su [http://localhost:8888](http://localhost:8888)

### Alternative use (no-copy):

Build docker image
```
docker build -t koda -f Dockerfile-no-copy .
```

Run image mounting current directory to /usr/src/app/koda

```
docker run --mount type=bind,source="$(pwd)",target=/usr/src/app -p 8888:8888 -it koda
```
