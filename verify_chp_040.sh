#!/bin/sh
FILENAME="/home/castlehoney/repos/course/dl/stanford-stages/chp040.edf"

# Removes both pickle files:
# rm /Users/unknown/data/sleep/narcoTest/CHP040.*pkl

# Remove previously pickled edf encoding data - useful when trying out different filters
# rm /Users/unknown/data/sleep/narcoTest/CHP040.pkl

# Remove previously picked hypnodensity data - useful when trying different models
# rm /Users/unknown/data/sleep/narcoTest/CHP040.hypno_pkl

python3 -W ignore inf_narco_app.py "$FILENAME" \
  '{"channel_indices":{"centrals":[3,4],"occipitals":[5,6],"eog_l":7,"eog_r":8,"chin_emg":9},
   "show":{"plot":false,"hypnodensity":false,"hypnogram":false,"diagnosis":true},
   "save":{"plot":true,"hypnodensity":true, "hypnogram":true,"diagnosis":true}
   }'
