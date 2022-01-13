# JSON configuration

The ___stanford-stages app___ may be run using __run_stanford_stages.py__ (or one of its wrapper scripts) with a Javascript object notation (json) configuration file, or by invoking the main method of __inf_narco_app.py__ module with a json file or passing json parameters to it directly.

The __run_stanford_stages.py__ provides additional functionality for configuring and using the ___stanford-stages app___ within a processing pipeline. As such, the json parameters are slightly different between __run_stanford_stages.py__ and __inf_narco_app.py__.  For example, the `channel_labels` field is used to list possible channel labels (e.g. 'C3-M1') that __run_stanford_stages.py__ will search  .edf file being processed for the corresponding channel index (e.g. 0) which is then updated to the corresponding `channel_indices` field values used by __inf_narco_app.py__.

A description of the supported key-value pairs for these files are provided here.  A sample json input file that can be used as a starting
point for __run_stanford_stages.py__ is included [below]() and in the repository's code base as well (see [stanford_stages.json]()).

# Key-value pairs

This section describes the json input file format used by .json configuration.

## `edf_pathname`

A valid directory name that will be searched for .EDF files.  All .edf files found in this path will be processed by the stanford-stages app.  This field takes priority over the `edf_filename` and `edf_files` fields.

## `edf_filename`

A valid filename (i.e. with the full path) of the .edf file that will be processed by the stanford-stages app.  This field takes priority over the `edf_files` field.  This field is ignored if `edf_pathname` has an entry.

## `edf_files`

A list of valid filenames (i.e. with the full path) of the .edf files that will be processed by the stanford-stages app.  This is useful for processing a subset of .edf files in the same folder or spread across different folders.  This field is ignored if `edf_pathname` or `edf_file` have entries.
 
## `channel_labels`

The `channel_labels` field contains keys for identifying channel labels, as specified in the .edf file's header section, for the following categories.  Multiple channel labels can be listed for each category by placing the labels with in square brackets (e.g. ['c3-m2', 'c3-m1']).  In the event that more than one label is found for a category, the first label is selected.  

* `central3`: EEG C-3 channel label(s)
* `central4`:  EEG C-4 channel label(s)
* `occipital1`: EEG O-1 channel label(s)
* `occipital2`: EEG O-2 channel label(s
* `eog_l`: Left EOG channel label(s)
* `eog_r`: Right EOG channel label(s)
* `chin_emg`: Chin EMG channel label(s)

Both EOG channels, a chin EMG channel, and at least one central and one occipital EEG channel is required.  When both `central3` and `central4` channel labels are provided the software will select the one best quality signal for the central EEG input. 
 Likewise, when both `occipital1` and `occipital2` channel labels are provided, the software will select the channel with highest quality signal for the the same is true for the occipital EEG input.
   

## `inf_config`

These settings override default configurations that are found in the __inf_config.py__ file.  The location of model paths must be updated with the location of the model files provided through the external links on the main readme page and which must be downloaded and extracted for use.
  
* ### `hypnodensity_model_root_path`

  The location of the path which holds the hypnodensity models used for scoring sleep stages.  Download these models from the links located on in the repository's readme file.

* ### `narcolepsy_classifier_path`

  The location of the narcolepsy classification models which are used to provide a likelihood of narcolepsy.  Currently, narcolepy classification models are only provided for the _manuscript branch_.  These models may be downloaded from the external links provided in the repository's readme file.

* ### `hypnodensity_scale_path`

  The location of the scaling files provided with the repository.  These files are provided as a zip archive (__scaling.zip__).  This file is included as a zip archive (__scaling.zip__) the repository's `ml/` subdirectory, which may be extracted in place to the `ml/scaling` directory.  The default behavior (i.e. if the field is not included) is to use `ml/scaling/` as a relative path of the calling method.  If you are calling the stanford-stages app from another location then it is necessary to specify the absolute location of the scaling directory.
  
* ### `psg_noise_filename` 

  This is the full filename (i.e. include the directory name) of __noise.mat__ file.  This file is included with the repository in the `ml/` subdirectory.  The default is to use this file at this location, relative to the calling method.  If you are calling the stanford-stages app from another location then it is necessary to specify the absolute location of the __noiseM.mat__ file.

* ### `lights_off`

  Indicates time elapsed (seconds) from the start of the study until _lights off_.  This identifies when the study starts and what data will be used by the models.  
Hypnodensity values prior to lights off will be scored 'nan' and hypnogram values prior to lights off will be scored as '7' (unknown).  The default value used in case
of an empty or non-existent entry is 0 (i.e. the study starts immediately).
 
* ### `lights_on`

  Indicates time elapsed (seconds) from the start of the study until _lights on.  This identifies when the study end.  Hypnodensity and hypnogram values for data after lights on will be scored as nan and '7' (unknown), respectively.  Negative values are allowed, and indicate elapsed time from the end of the study until lights on.  For example, -60 indicates lights on occurs 1 minute before the recording ends.  A value of 0 indicates that the entire study, from lights off, is used.  The default value used in case of an empty or non-existent entry is 0.

* ### `models_used`

  A list of the models to be used.  There are 16 models provided for scoring sleep and also for classifying narcolepsy (note: narcolepy classification models are only for the _manuscript branch_). Performance is best generalized by using all 16 models, however, using fewer models is faster.  The models are identified using the template __ac_rh_ls_lstm\__\<dd\>___ where \<dd\> is the 0 paded two digit model number (i.e. "__ac_rh_ls_lstm_01__" through "__ac_rh_ls_lstm_16__").


* ### `atonce`

  Not implemented yet.

* ### `segsize`

  Not implemented yet.  
  
## `output_path`

The output path specifies the location that selected files are saved to by the stanford-stages app.  These output files are identified using the `save` field.

## `save`

The `save` field identifies intermediate and output data to save.  Saving intermediate can help speed up future processing but may take up significant storage space; the encoding files can be particularly large (e.g. approximately 100 MB per .edf study duration)   

The following keys take hold boolean (true/false) values to indicate which files will be saved.  All fields are taken to be 'True' by default.

__\[Default\]__ values are listed in square brackets next to each field.  These specify the application's behavior when the field is excluded. 
A description of the behavior of each field is given below when the value is 'True'.  If the value is 'False' then no output file is created. 
Output files are named by replacing the '.edf' suffix of the study name with the suffix identified for each field below.  

* ### `diagnosis` \[True\]

  When 'True', a text file is created containing the narcolepsy score, which ranges between -1 and 1, and the narcolepsy classification.  The file suffix is '.diagnosis.txt'.  For example, diagnosis save file for 'CHP040.edf' would be 'CHP040.diagnosis.txt'.
  
* ### `encoding`  \[True\]

  This contains the cross-correlation encodings of the selected edf channels.  It is saved as an .h5 file in the Hierarchical Data Format (HDF).
    The file suffix is '.h5'.

* ### `hypnodensity` \[True\]

  Saves the hypnodensity in 15 second epochs in multiple formats.  The hypnodensities generated for each model applied are saved as an HDF file using '.hypnodensity.h5' as the file suffix, and as a Python pickle file by using the .edf file and '.hypnodensity.pkl' as the file suffix.
  An aggregated hypnodensity (i.e. average of all models applied) is saved as a comma-separated-values (csv) using '.hypnodensity.txt' as the filename suffix.

* ### `hypnogram` \[True\]

  Saves a hypnogram of the study in 15 second epochs.  The file is saved in text format (UTF-8) using the base name of the .edf file and using '.txt' as the file suffix.
  
* ### `hypnogram_30_sec`  \[True\]

  Saves a hypnogram of the study in 30 second epochs.  The file is saved in text format (UTF-8) using the base name of the .edf file and using '.STA' as the file suffix.
  
* ### `plot` \[True\]

  Saves a plot of the hypnodensity as a PNG image using the file suffix '.hypnodensity.png'.  

## `show`

The `show` field identifies which results are output to the console window (e.g. terminal, shell, or Python interpreter) when running the stanford-stages app.    

The following keys use boolean values (true/false) to indicate if output is shown or not for the associated category.  

Any field not specified in the JSON file will be treated as 'False' and not shown.  

* ### `diagnosis`

  Display the narcolepsy score and associated diagnosis.

* ### `hypnodensity`

  Display consolidated hypnodensity values on 15 s epochs, which have been consolidated by averaging the results of all models (i.e. up to 16 models)).

* ### `hypnogram`

  Displays the sleep stages on a 15 s epoch as derived from the consolidated hypnodensitiy.  

* ### `hypnogram_30_sec`

  Displays the sleep stages on a 30 s epoch as derived from the consolidated hypnodensitiy.

* ### `plot` 

  Shows a visual representation of the hypnodensity in a stand alone window.  The app will wait until the user closes this window before continuing, which be may be undesired when running app as a pipeline to process multiple studies at a time.
   
# Sample JSON content for _run_stanford_stages.py_ 
```json
{
    "channel_labels": {
        "central3": "EEG C3-A2",
        "central4": "EEG C4-A1",
        "chin_emg": "EMG Chin",
        "eog_l": "EOG LOC-A2",
        "eog_r": "EOG ROC-A2",
        "occipital1": "EEG O1-A2",
        "occipital2": "EEG O2-A1"
    },
    "edf_filename": "C:/Data/ml/CHP040.edf",
    "inf_config": {
        "atonce": "1000",
        "hypnodensity_model_dir": "F:/ml/ac/",
        "hypnodensity_scale_path": "F:/ml/scaling",
        "lights_filename": "",
        "lights_off": "",
        "lights_on": "",
        "models_used": [
            "ac_rh_ls_lstm_01",
            "ac_rh_ls_lstm_02",
            "ac_rh_ls_lstm_03",
            "ac_rh_ls_lstm_04",
            "ac_rh_ls_lstm_05",
            "ac_rh_ls_lstm_06",
            "ac_rh_ls_lstm_07",
            "ac_rh_ls_lstm_08",
            "ac_rh_ls_lstm_09",
            "ac_rh_ls_lstm_10",
            "ac_rh_ls_lstm_11",
            "ac_rh_ls_lstm_12",
            "ac_rh_ls_lstm_13",
            "ac_rh_ls_lstm_14",
            "ac_rh_ls_lstm_15",
            "ac_rh_ls_lstm_16"
        ],
        "narco_classifier_path": "C:/Data/OAK/narco_test_updated_b_gp_models_1000",
        "psg_noise_filename": "F:/ml/noiseM.mat",
        "segsize": "60"
    },
    "output_path": "C:/Data/ml/results",
    "save": {
        "diagnosis": true,
        "encoding": true,
        "hypnodensity": false,
        "hypnogram": false,
        "hypnogram_30_sec": false,
        "plot": false
    },
    "show": {
        "diagnosis": true,
        "hypnodensity": false,
        "hypnogram": false,
        "hypnogram_30_sec": true,
        "plot": false
    }
}
```    

# JSON configuration for inf_narco_app

This section describes the JSON key value pairs which can be passed directly to __inf_narco_app.py__ for developers who want to create their own wrapper functionality in liue of the __run_stanford_stages.py__.  

## `channel_indices`
   
### Description
 Assigns channel indices corresponding to the central, occipital, ocular, and chin EMG. 
 
 One or two EEG channels may be assigned to the central and occipital categories.  Both left and right EOG channels are required for the corresponding occular category, and one channel is required for the chin EMG.  In the event that two channels are presented for an EEG category (central or occipital), a quality metric is calculated for each channel, and the optimal channel is selected for use.  Channel indices are 0 based and correspond to the channel labels list provided in the .EDF file header.

 ### Supported keys and values

* `central`: C3 or C4 channel index
* `central3`: C3 channel index
* `central4`:  C4 channel index
* `centrals`: \[C3 channel index, C4 channel index\]
* `occipital`: O1 or O2 channel index
* `occipital1`: O1 channel index
* `occipital2`: O2 channel index
* `occipitals`: \[O1 channel index, O2 channel index\]
* `eog_l`: Left EOG channel index
* `eog_r`: Right EOG channel index
* `eogs`: \[Left EOG channel index, Right EOG channel index\]
* `chin_emg`: EMG channel index

## `show`
Flags for determining which results are output to the console.

### Supported keys and [default] values
* `plot`: true or [false]
* `hypnodensity`: true or [false]
* `hypnogram`: true or [false]
* `diagnosis`: true or [false]

## `save`

Flags for determining which results are saved to disk.  Files are saved to the same directory as the input .edf file.  Save filenames are generated by pairing the .edf file's basename with the file extension matching the output type requested.

### Supported keys and [default] values

* `plot`: [true] or false  (.hypnodensity.png)
* `encoding`: [true] or false (.h5)
* `hypnodensity`: [true] or false (.hypnodensity.txt)
* `hypnogram`: [true] or false (.hypnogram.txt)
* `diagnosis`: [true] or false (.diagnosis.txt)

[default] values (file extension used by default when true)

## `filename`

Alternative filenames to use in place of the default naming convention and path location used when saving results.  

### Supported keys and values

* `plot`: Full path for hypnodensity image file
* `hypnodensity`: Full path for hypnodensity output text file
* `hypnogram`: Full path for hypnogram output text file
* `diagnosis`: Full path for diagnosis output text file
