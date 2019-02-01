# Stanford Stages

Automated sleep staging scoring and narcolepsy identification.

## Instructions

Software requirements and instructions can be found [here](
https://docs.google.com/document/d/e/2PACX-1vTvin7Gdn7FN9-2NbAQKgnApR6F73en46cTFYosxCMlgFgp3pMqSJgDthaCghrfAGIZ_BoKThb4bHtt/pub).


# Initial configuration

## Models

Classification models are hosted externally and should be downloaded and extracted into the repositories 'ml/' subfolder

* ac.zip - www.informaton.org/narco/ml/ac.zip [1.0 GB]
* gp.zip - www.informaton.org/narco/ml/gp.zip  [64 MB]

When complete the 'ml/' directory tree should like:<pre>
ac/
gp/
noiseM.mat
scaling</pre>

## Validation

The sleep study CHP_040.edf is may be used to verify your setup.  It can be downloaded from the following mirrors:

1. [Mirror 1](https://stanfordmedicine.box.com/shared/static/0lvvyaprzinzz7dult87t7hr96s2dnqq.edf) [380 MB]
2. [Mirror 2](https://www.informaton.org/narco/ml/validation/CHP_040.edf) [380 MB]

The sleep study may be placed in any directory.  Edit the shell script <i>verify_chp_040.sh</i> so that the
absolute pathname for CHP040.edf is given for the FILENAME variable.  In this example, it is assumed to have been saved
to the directory `/Users/unknown/data/sleep/narcoTest/` and so the the FILENAME variable should be set as follows:

<pre><code>FILENAME="/Users/unknown/data/sleep/narcoTest/CHP040.edf"</code></pre>

See the single model validation configuration

### Single model validation

This validation tests your configuration using one model, 'ac_rh_ls_lstm_01' and is recommended as
a first step to making sure the application is configured correctly simply because it takes
less time to check than running all 16 models.  

1. Edit the inf_config.py file so that the model_used property is set as follows:

<pre><code>self.models_used = ['ac_rh_ls_lstm_01']</code></pre>

2. Run the shell script from a command line terminal as:

<pre><code>sh ./verify_chp_040.sh</code></pre>

3. Check results

Upon successful completion, a hypnogram file and hypnodensity image will be created
and saved in the same directory as the input CHP040.edf file.  

Expected results for the ac_rh_ls_lstm_01 hypnogram and hypnodensity results can be found here:

* [Hypnogram (single model)](https://www.informaton.org/narco/ml/validation/ac_rh_ls_lstm_01/CHP_040.hypnogram.txt) [270 KB]

* [Hypnodensity image (single model)](https://www.informaton.org/narco/ml/validation/ac_rh_ls_lstm_01/CHP_040.hypnodensity.png) [155 KB]

Expected diagnosis output is:

<pre>Score: -0.0076
Diagnosis: Narcolepsy type 1</pre>


### All

This validation tests your configuration using all 16 models and is the recommended way
for running the application.  

1. Edit the inf_config.py file and ensure that model_used property is set as follows:

<pre><code>self.models_used = ['ac_rh_ls_lstm_01', 'ac_rh_ls_lstm_02',
                    'ac_rh_ls_lstm_03', 'ac_rh_ls_lstm_04',
                    'ac_rh_ls_lstm_05', 'ac_rh_ls_lstm_06',
                    'ac_rh_ls_lstm_07', 'ac_rh_ls_lstm_08',
                    'ac_rh_ls_lstm_09', 'ac_rh_ls_lstm_10',
                    'ac_rh_ls_lstm_11', 'ac_rh_ls_lstm_12',
                    'ac_rh_ls_lstm_13', 'ac_rh_ls_lstm_14',
                    'ac_rh_ls_lstm_15', 'ac_rh_ls_lstm_16']</code></pre>

1. Run the shell script from a command line terminal as:

<pre><code>sh ./verify_chp_040.sh</code></pre>

Note: If the shell script has been run before using a single model, you will need to delete the previously cached "pickle" files.  Edit the verify_chp_040.sh script and uncomment the line:

<pre><code># rm /Users/unknown/data/sleep/narcoTest/CHP040.\*pkl</code></pre>
to remove any previously saved pickle files.

2. Check results

Upon successful completion, a hypnogram file and hypnodensity image will be created
and saved in the same directory as the input CHP040.edf file.  

Expected results for may be found here:

* [Hypnogram (full)](https://www.informaton.org/narco/ml/validation/all/CHP040.hypnogram.txt) [270 KB]

* [Hypnodensity image (full)](https://www.informaton.org/narco/ml/validation/all/CHP040.hypnodensity.png) [155 KB]

Expected diagnosis output is:

<pre>Score: 0.1658
Diagnosis: Narcolepsy type 1</pre>

## Output

### Narcolepsy diagnosis

The algorithm produces values between −1 and 1, with 1 indicating a high probability of narcolepsy. The cut-off threshold between narcolepsy type 1 and “other“ is set at −0.03.  See [Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy](https://www.nature.com/articles/s41467-018-07229-3) for details.  

## Input

### JSON arguments

Javascript object notation (json) is used for passing parameters to the application.  

Three parameters that can be adjusted include setting the channel indices for input,
and what output should be printed to the screen and/or saved to disk.  
The keys for these parameters, and their corresponding definitions are as follows:

* `channel_indices`
   * Description
    Assigns channel indices corresponding to the central, occipital, occular, and chin EMG. One or two EEG channels may be assigned
to the central and occipital categories.  Both left and right EOG channels are required for the corresponding occular category, and one channel is required for the chin EMG.  In the event that two channels are presented for an EEG category (central or occipital), a quality metric is calculated
for each channel, and the optimal channel is selected for use.  Channel indices are 0 based and correspond to the channel labels list provided in
the .EDF file header.
   * Supported keys and values  
       * `central`: C3 or C4 channel index
       * `central3`: C3 channel index
       * `central4`:  C4 channel index
       * `centrals`: [C3 channel index, C4 channel index]
       * `occipital`: O1 or O2 channel index
       * `occipital1`: O1 channel index
       * `occipital2`: O2 channel index
       * `occipitals`: [O1 channel index, O2 channel index]
       * `eog_l`: Left EOG channel index
       * `eog_r`: Right EOG channel index
       * `eogs`: [Left EOG channel index, Right EOG channel index]
       * `chin_emg`: EMG channel index
* `show`
   * Description
    Flags for determining which results are output to the console.
   * Supported keys and [default] values
      * `plot`: true or [false]
      * `hypnodensity`: true or [false]
      * `hypnogram`: true or [false]
      * `diagnosis`: [true] or false

* `save`
   * Description
    Flags for determining which results are saved to disk.  Files are saved to the same directory
    as the input .edf file.  Save filenames are generated by pairing the .edf file's basename with
    the file extension matching the output type requested.
   * Supported keys : [default] values (file extension)
      * `plot`: [true] or false  (.hypnodensity.png)
      * `hypnodensity`: [true] or false (.hypnodensity.txt)
      * `hypnogram`: [true] or false (.hypnogram.txt)
      * `diagnosis`: [true] or false (.diagnosis.txt)

* `filename`
   * Description
    Alternative filename to use in place of the default naming convention and path location used when saving results.  
   * Supported keys
      * `plot`: Full path for hypnodensity image file.
      * `hypnodensity`: Full path for hypnodensity output text file
      * `hypnogram`: Full path for hypnogram output text file
      * `diagnosis`: Full path for diagnosis output text file
