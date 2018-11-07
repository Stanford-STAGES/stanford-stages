#!/bin/sh
FILENAME="/Users/unknown/data/narco_test/SSC_7337_1.EDF"

python3 -W ignore inf_narco_app.py "$FILENAME" \
  '{"channel_indices":{"central":16,"occipital":27,"eog_l":26,"eog_r":30,"chin_emg":0}, "hypnodensity":{"showplot":true}}'


