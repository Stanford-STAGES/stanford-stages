#!/bin/sh
FILENAME="/Users/unknown/data/sleep/narcoTest/CHP040.edf"

# rm /Users/unknown/data/sleep/narcoTest/CHP040.*pkl

python3 -W ignore inf_narco_app.py "$FILENAME" \
  '{"channel_indices":{"central":3,"occipitals":[5,6],"eog_l":7,"eog_r":8,"chin_emg":9},
   "show":{"plot":false,"hypnodensity":false,"hypnogram":false},
   "save":{"plot":false,"hypnodensity":true, "hypnogram":true}
   }'
