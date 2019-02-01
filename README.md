# Stanford Stages

Automated sleep staging scoring and narcolepsy identification.

## Instructions

Software requirements and instructions can be found [here](
https://docs.google.com/document/d/e/2PACX-1vTvin7Gdn7FN9-2NbAQKgnApR6F73en46cTFYosxCMlgFgp3pMqSJgDthaCghrfAGIZ_BoKThb4bHtt/pub).

### JSON arguments

# Initial configuration

## Models

Classification models are hosted externally and should be downloaded and extracted into the repositories 'ml/' subfolder

* ac.zip - www.informaton.org/narco/ml/ac.zip [1.0 GB]
* gp.zip - www.informaton.org/narco/ml/gp.zip  [64 MB]

When complete the 'ml/' directory tree should like this:

<pre><code>
ac/
gp/
noiseM.mat
scaling
</code></pre>

## Validation

This validation tests your configuration using one model, 'ac_rh_ls_lstm_01'.
Edit the inf_config.py file so that the model_used property is set as follows:

<pre><code>self.models_used = ['ac_rh_ls_lstm_01']</code></pre>

The sleep study CHP_040.edf is may be used to verify your setup.  It can be downloaded from the following mirrors:

1. [Mirror 1](https://stanfordmedicine.box.com/shared/static/0lvvyaprzinzz7dult87t7hr96s2dnqq.edf) [380 MB]
2. [Mirror 2](https://www.informaton.org/narco/ml/validation/CHP_040.edf) [380 MB]

The sleep study may be placed in any directory.  Edit the shell script <i>verify_chp_040.sh</i> so that the
absolute pathname for CHP040.edf is given for the FILENAME variable.  In this example, it is assumed to have been saved
to the directory "/Users/unknown/data/sleep/narcoTest/" and so the the FILENAME variable should be set as follows:

FILENAME="/Users/unknown/data/sleep/narcoTest/CHP040.edf"

Run the shell script from a command line terminal as:

<pre><code>sh ./verify_chp_040.sh</code></pre>

Upon successful completion, a hypnogram file and hypnodensity image will be created
and saved in the same directory as the input CHP040.edf file, that is "/Users/unknown/data/sleep/narcoTest/".

Expected results for the hypnogram and hypnodensity can be found here:

* [Hypnogram](https://www.informaton.org/narco/ml/validation/CHP_040.hypnogram) [270 KB]

* [Hypnodensity image](https://www.informaton.org/narco/ml/validation/CHP_040.hypnodensity.png) [155 KB]

* Diagnosis

Score: -0.0076, Diagnosis: Narcolepsy type 1

### Narcolepsy output

The algorithm produces values between −1 and 1, with 1 indicating a high probability of narcolepsy. The cut-off threshold between narcolepsy type 1 and “other“ is set at −0.03.  See [Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy](https://www.nature.com/articles/s41467-018-07229-3) for details.  
