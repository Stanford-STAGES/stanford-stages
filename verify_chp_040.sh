#!/bin/sh
FILENAME="/Users/unknown/data/sleep/CHP040.edf"
rm /Users/unknown/data/sleep/CHP040.*pkl

python3 -W ignore inf_narco_app.py "$FILENAME" \
  '{"channel_indices":{"centrals":[3,4],"occipitals":[5,6],"eog_l":7,"eog_r":8,"chin_emg":9}, "hypnodensity":{"showplot":true}}'
