SETLOCAL
SET FILENAME="C:\Data\CHP040.edf"
dir %FILENAME%

:: Removes both pickle files:
:: del C:\Data\CHP040.*pkl

:: Remove previously pickled edf encoding data - useful when trying out different filters
:: del C:\Data\CHP040.pkl

:: Remove previously picked hypnodensity data - useful when trying different models
:: del C:\Data\CHP040.hypno_pkl

python inf_narco_app.py %FILENAME% ^
"{\"channel_indices\":{\"centrals\":[3,4], \"occipitals\":[5,6],\"eog_l\":7,\"eog_r\":8,\"chin_emg\":9}, ^
\"show\":{\"plot\":false,\"hypnodensity\":false,\"hypnogram\":false,\"diagnosis\":true}, ^
\"save\":{\"plot\":true,\"hypnodensity\":true, \"hypnogram\":true,\"diagnosis\":true}}"
