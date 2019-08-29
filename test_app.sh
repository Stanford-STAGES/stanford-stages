#!/bin/sh
# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore edfReport.py Demo.EDF '{"channel_labels":{"central":"Chin","occipital":"EKG","eog_l":"C3-M2","eog_r":"C3-M2","chin_emg":"EKG"},"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3},"lightsOn":1152,"lightsOff":33}'
DEMONAME="Demo.EDF"
PATHNAME="/Users/unknown/data/narco_test/"
FILENAME="C9188_4 135030"
EXT=".EDF"
FULLFILE="${PATHNAME}${FILENAME}${EXT}"
CLEANUP=true
CLEANUP=false
CACHEFILES=${PATHNAME}$FILENAME.*pkl
FILES="/Users/unknown/data/narco_test/C9188_4\ 135030.*pkl"

#FULLFILE=${PATHNAME}C9${EXT}

# Minimal example
#/Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore inf_narco_app.py Demo.EDF '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}}'

# Show plot
#python3 -W ignore inf_narco_app.py Demo.EDF\
#  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}, "hypnodensity":{"showplot":true}}'

# No plot
#python3 -W ignore inf_narco_app.py Demo.EDF \
#  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3}, "hypnodensity":{"showplot":false}}'

# Removes both pickle files:
#if [$ -eq 1]
if $CLEANUP
then
  echo Removing previously cached \(pickled\) files.
  echo "rm ${CACHEFILES}"
  #rm ${CACHEFILES}
  # Failes:  ls "/Users/unknown/data/narco_test/C9188_4\ 135030.*pkl"
  # Works:   ls /Users/unknown/data/narco_test/C9188_4\ 135030.*pkl
  #ls -- "${CACHEFILES}"
  #ls -- "$CACHEFILES"
  #ls "${CACHEFILES}"
  # rm "$FILES"
  # ls "${FULLFILE}"
fi
# rm /Users/unknown/data/sleep/narcoTest/CHP040.*pkl

# Remove previously pickled edf encoding data - useful when trying out different filters
# rm /Users/unknown/data/sleep/narcoTest/CHP040.pkl

# Remove previously picked hypnodensity data - useful when trying different models
# rm /Users/unknown/data/sleep/narcoTest/CHP040.hypno_pkl

python3 -W ignore inf_narco_app.py "$FULLFILE" \
  '{"channel_indices":{"central3":1,"central4":2,"occipital":3,"eog_l":0,"eog_r":1,"chin_emg":4}, "hypnodensity":{"showplot":true}}'

python3 -W ignore inf_narco_app.py "$FULLFILE" \
  '{"channel_indices":{"central3":2,"occipital2":3,"eog_l":0,"eog_r":1,"chin_emg":4},"show":{"plot":false,"hypnodensity":false,"hypnogram":false,"diagnosis":true},"save":{"plot":true,"hypnodensity":true, "hypnogram":true,"diagnosis":true}}'

#python3 -W ignore inf_narco_app.py "/Users/unknown/data/narco_test/C9188_4 135030.EDF" \
#  '{"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":1,"chin_emg":4}, "hypnodensity":{"showplot":true}}'


# /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -W ignore inf_narco_app.py Demo.EDF '{"channel_labels":{"central":"Chin","occipital":"EKG","eog_l":"C3-M2","eog_r":"C3-M2","chin_emg":"EKG"},"channel_indices":{"central":2,"occipital":3,"eog_l":0,"eog_r":0,"chin_emg":3},"lightsOn":1152,"lightsOff":33}'
