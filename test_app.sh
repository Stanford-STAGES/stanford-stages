#!/bin/sh
# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore edfReport.py Demo.EDF '{"channel_labels":{"central":"Chin","occipital":"EKG","eog_l":"C3-M2","eog_r":"C3-M2","chin_emg":"EKG"},"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3},"lightsOn":1152,"lightsOff":33}'
DEMONAME="Demo.EDF"
FILENAME="~/data/narco_test/C9188_4\ 135030.EDF"
# Minimal example
#/Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore inf_narco_app.py Demo.EDF '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}}'

# Show plot
#python3 -W ignore inf_narco_app.py Demo.EDF\
#  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}, "hypnodensity":{"showplot":true}}'

# No plot
#python3 -W ignore inf_narco_app.py Demo.EDF \
#  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}, "hypnodensity":{"showplot":false}}'

python3 -W ignore inf_narco_app.py "/Users/unknown/data/narco_test/C9188_4 135030.EDF" \
  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":1,"chin_emg":4}, "hypnodensity":{"showplot":true}}'


# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore inf_narco_app.py Demo.EDF '{"channel_labels":{"central":"Chin","occipital":"EKG","eog_l":"C3-M2","eog_r":"C3-M2","chin_emg":"EKG"},"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3},"lightsOn":1152,"lightsOff":33}'
