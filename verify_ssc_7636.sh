#!/bin/sh
FILENAME="/Users/unknown/data/narco_test/SSC_7636_1.EDF"

python3 -W ignore inf_narco_app.py "$FILENAME" \
  '{"channel_indices":{"central":23,"occipital":34,"eog_l":33,"eog_r":37,"chin_emg":0}, "hypnodensity":{"showplot":true}}'


