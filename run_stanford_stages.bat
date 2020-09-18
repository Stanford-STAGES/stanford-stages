SETLOCAL
SET FILENAME="C:\Data\ml\CHP040.edf"
SET CONFIG="C:\Data\ml\signal_labels.json"
dir %FILENAME%

:: Removes both pickle files:
:: del C:\Data\CHP040.*pkl

:: Remove previously pickled edf encoding data - useful when trying out different filters
:: del C:\Data\CHP040.pkl

:: Remove previously picked hypnodensity data - useful when trying different models
:: del C:\Data\CHP040.hypno_pkl

python run_stanford_stages.py %FILENAME% %CONFIG%