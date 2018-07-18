#!/bin/sh
# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore edfReport.py Demo.EDF '{"channel_labels":{"central":"Chin","occipital":"EKG","eog_l":"C3-M2","eog_r":"C3-M2","chin_emg":"EKG"},"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3},"lightsOn":1152,"lightsOff":33}'

# Minimal example
#/Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore inf_narco_app.py Demo.EDF '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}}'

# Show plot
python3 -W ignore inf_narco_app.py Demo.EDF\
  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}, "hypnodensity":{"showplot":true}}'

#python3 -W ignore inf_narco_app.py Demo.EDF \
#  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}, "hypnodensity":{"showplot":true}}'


# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore inf_narco_app.py Demo.EDF '{"channel_labels":{"central":"Chin","occipital":"EKG","eog_l":"C3-M2","eog_r":"C3-M2","chin_emg":"EKG"},"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3},"lightsOn":1152,"lightsOff":33}'
