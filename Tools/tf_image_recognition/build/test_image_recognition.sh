#!/bin/bash

# inception_v4
DATA_DIR=../data/car1.jpg
MODEL_DIR=../data
./image_recognition \
  --image=${DATA_DIR} \
  --graph=${MODEL_DIR}/inception_v4_zh_frozen_tool.pb \
  --labels=${MODEL_DIR}/auto_metadata.txt

