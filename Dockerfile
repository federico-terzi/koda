FROM python:3.7-buster

RUN apt-get update \
    && apt-get -y install \
    build-essential \
    tesseract-ocr 

#RUN apk update \
#    && apk add \
#    build-base \
#    tesseract-ocr \
#    tesseract-ocr-dev

RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN python -m pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1

COPY . .
    
