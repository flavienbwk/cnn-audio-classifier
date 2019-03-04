FROM ubuntu:xenial

LABEL author Flavien Berwick "flavien@berwick.fr"
WORKDIR /root

RUN apt-get update -y

RUN apt-get install ffmpeg -y
RUN apt-get install python -y
RUN apt-get install python-pip -y
RUN apt-get install python-tk -y
RUN apt-get install python-opencv -y

RUN pip install 'matplotlib<3'
RUN pip install tensorflow
RUN pip install keras
RUN pip install librosa
RUN pip install numpy
RUN pip install pandas
RUN pip install opencv-python
RUN pip install opencv-contrib-python
