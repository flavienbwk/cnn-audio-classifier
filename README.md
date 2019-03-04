# Convolutional Neural Network - Audio Classifier

Based on the great work of [Ajay Halthor](https://github.com/ajhalthor/audio-classifier-convNet/blob/master/env_sound_discrimination.ipynb), this software will allow you to get a JSON list of all the sounds labels detected at a specific period on a `.wav` record.

## Architecture

- learning.py
  - Learns from the UrbanSound8K.csv file.
  - Generates a "data.cnn" file containing the binary representation of all the data learned.
- process.py
  - Processes the record, looking for `sample.wav` at the root of the project.
  - Generates the JSON file containing the sounds labels at a specific period.

## Learning

The software uses the [UrbanSound8K dataset, that you can download here](https://urbansounddataset.weebly.com/urbansound8k.html) (~6 Go <=> 8732 sounds). Please put it at the root of the project.

## Launch the project 

### With Docker

Automatically loads all dependencies. May be long to launch as the UrbanSound8K dataset is big.

1. Run `docker-compose up -d`
2. Access the container via `docker exec -it <container_id> bash`
   1. To see the list of the containers, run `docker ps`

### Without Docker

1. Install the dependencies you can see in [./Dockerfile](./Dockerfile)
2. Run `python environment.py`