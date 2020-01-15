# koda
> Deep Learning-enhanced Keyword Detection in Document Images

*koda* was created by Matteo Pellegrino and Federico Terzi for the
Computer Vision course held by Professor Di Stefano.

For an indepth explanation, check out the [paper](/paper/paper.pdf).

It consists of a pipeline to analyze document images and extracting/highlighting
specific keywords, such as shown in this image:

![koda example](/paper/images/result.png)

A combination of Deep learning-based edge detection, Hough transforms, OCR and
warping was used to achieve the result.

![koda pipeline](/paper/images/ipt.png)


# Usage

Start by building the docker image with (it may require using sudo on linux):

```
sudo ./build_docker.sh
```

Then you can either start the Flask server with:

```
sudo ./start_server.sh
```

Or you can open the Jupyter notebook with:

```
sudo ./start_jupyter.sh
```

And then navigate to [http://localhost:8888](http://localhost:8888)
