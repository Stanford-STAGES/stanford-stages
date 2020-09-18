# Stanford Stages

An automated sleep stage scoring and narcolepsy identification software program written in Python.

The stanford-stages app uses machine learning to perform automated sleep stage scoring and narcolepsy identification of nocturnal polysomnography (PSG) studies, which must be in European Data Format (.edf).
The software requires previously trained models to accomplish this as described in the 2018 manuscript ["Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy."](https://www.nature.com/articles/s41467-018-07229-3)
## Branches

Git provides support for multiple _branches_ of development.  Notable branches in the stanford-stages repository include:

1. __Master__ 

   The ___master branch___ provides GPU (Nvidia) compatibility and supports TensorFlow 2.0 and GPflow 2.0.  Source code from this branch can be used to automatically score sleep stages using the original hypnodensity models (ac.zip) included below.  It is _not_ compatible with the previously trained narcolepsy classification models.  We are working on developing new classification models.

1. __Manuscript__ 

   The ___manuscript branch___ includes the code presented with the 2018 manuscript.  It uses earlier versions of GPflow and TensorFlow, to retain compatibility with the narcolepsy classification models (gp.zip) presented with the manuscript.  
  
    See the [__manuscript branch__ README page](https://github.com/Stanford-STAGES/stanford-stages/blob/master/Manuscript_Branch_README.md) for configuration instructions using this branch and links to the narcolepsy models.  

1. __Beta__ 

   The ___beta branch___ serves as development branch for making and testing changes to the ___master branch___ and is not always stable.      


   **NOTE**: Only Python 3.6 and later is supported. Previous versions of Python have been shown to yield erroneous results. 

# Configuration

These instruction are for the master branch.  

## Models

The sleep staging classification models are hosted externally at the following location:

* ac.zip - www.informaton.org/narco/ml/ac.zip [770 MiB, 807 MB]
* <strike>gp.zip</strike>*


 
Download and extract the [ac.zip](www.informaton.org/narco/ml/ac.zip) file to your computer system.  These models, along with a PSG are necessary d by the software to run.
The .zip file may be deleted once its contents have been extracted.  
A Javascript object notation (json) file is used to configure the software.   
   

When complete the 'ml/' directory tree should like:<pre>
ac/
<strike>gp/</strike>
noiseM.mat
scaling</pre>

*_Note_ See _manuscript branch_ [readme](https://github.com/Stanford-STAGES/stanford-stages/blob/master/Manuscript_Branch_README.md) for links to the gp.zip file originally posted with the manuscript.  
 
## Validation

The sleep study CHP_040.edf is may be used to verify your setup.  It can be downloaded from the following mirrors:

1. [Mirror 1](https://stanfordmedicine.box.com/shared/static/0lvvyaprzinzz7dult87t7hr96s2dnqq.edf) [380 MB]
2. [Mirror 2](https://www.informaton.org/narco/ml/validation/CHP040.edf) [380 MB]

The sleep study may be placed in any directory.  Edit the shell script <i>verify_chp_040.sh</i> so that the absolute pathname for CHP040.edf is given for the FILENAME variable.  In this example, it is assumed to have been saved to the directory `/Users/unknown/data/sleep/narcoTest/` and so the the FILENAME variable should be set as follows:

<pre><code>FILENAME="/Users/unknown/data/sleep/narcoTest/CHP040.edf"</code></pre>

See the single model validation configuration

### Using one model (quick check)

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

* [Hypnogram (single model)](https://www.informaton.org/narco/ml/validation/ac_rh_ls_lstm_01/CHP040.hypnogram.txt) [4 KB]

* [Hypnodensity image (single model)](https://www.informaton.org/narco/ml/validation/ac_rh_ls_lstm_01/CHP040.hypnodensity.png) [158 KB]

Expected diagnosis output is:

<pre>Score: -0.0076
Diagnosis: Narcolepsy type 1</pre>


### Using all models

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

2. Run the shell script from a command line terminal as:

   <pre><code>sh ./verify_chp_040.sh</code></pre>

   Note: If the shell script has been run before using a single model, you will need to delete the previously cached "pickle" files.  Edit the verify_chp_040.sh script and uncomment the line:

   <pre><code># rm /Users/unknown/data/sleep/narcoTest/CHP040.\*pkl</code></pre>
   to remove any previously saved pickle files.

3. Check results

Upon successful completion, a hypnogram file and hypnodensity image will be created
and saved in the same directory as the input CHP040.edf file.  

Expected results for may be found here:

* [Hypnogram (full)](https://www.informaton.org/narco/ml/validation/all/CHP040.hypnogram.txt) [4 KB]

* [Hypnodensity image (full)](https://www.informaton.org/narco/ml/validation/all/CHP040.hypnodensity.png) [155 KB]

Expected diagnosis output is:

<pre>Score: 0.1658
Diagnosis: Narcolepsy type 1</pre>

## Output

### Narcolepsy diagnosis

The algorithm produces values between −1 and 1, with 1 indicating a high probability of narcolepsy. The cut-off threshold between narcolepsy type 1 and “other“ is set at −0.03.  See [Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy](https://www.nature.com/articles/s41467-018-07229-3) for details.  

### Hypnogram

The hypnogram provides a numeric indicator of wake or sleep stage for every epoch scored.  
By default, epochs are scored in 15 s intervals as follows:

* Wake: `0`
* Stage 1 sleep: `1`
* Stage 2 sleep: `2`
* Stage 3/4 sleep: `3`
* Rapid eye movement (REM) sleep: `5`

### Hypnodensity

#### Text
The hypnodensity text output is the same length (number of epochs/rows) as the hypnogram.  Instead of a
sleep score, however, five probabilities are given representing the likelihood of the
sleep stage corresponding to its column for the current epoch.  Probabilities are ordered
starting with wake, and moving to deeper stages of sleep, with REM sleep last.

For example:

<pre>0.24 0.01 0.02 0.03 0.70</pre>

Represents

* Wake: 24% likelihood
* Stage 1 sleep: 1% likelihood
* Stage 2 sleep: 2% likelihood
* Stage 3/4 sleep: 3% likelihood
* REM sleep: 70% likelihood

#### Image

The x-axis of the hypnodensity plot represents the epochs from the beginning to the
end of the study, which are measured in 15 s intervals by default.  The y-axis ranges from
0 to 1 and represents the hypnodensity - the probability of each sleep stage and wake.  
Different colors are used to represent wake and each sleep stage.  For each epoch, the probability of wake is drawn first as a vertical line to the proportion of the y-axis range it represents.
Stage 1 is then stacked on top of this, followed by Stages 2, 3/4, and REM sleep such that the
entire spectrum is covered.  Color matching is as follows:

* Wake: white
* Stage 1 sleep: pink
* Stage 2 sleep: aqua/turquoise
* Stage 3/4 sleep: blue
* REM sleep: green

## Input

### JSON arguments

Javascript object notation (json) is used for passing parameters to the application.  

Three parameters that can be adjusted include setting the channel indices for input,
and what output should be printed to the screen and/or saved to disk.  
The keys for these parameters, and their corresponding definitions are as follows:

* `channel_indices`
   * Description
   
    Assigns channel indices corresponding to the central, occipital, ocular, and chin EMG. One or two EEG channels may be assigned
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
   * Supported keys: allowed and [default] values
      * `plot`: true or [false]
      * `hypnodensity`: true or [false]
      * `hypnogram`: true or [false]
      * `diagnosis`: [true] or false

* `save`
   * Description
    Flags for determining which results are saved to disk.  Files are saved to the same directory
    as the input .edf file.  Save filenames are generated by pairing the .edf file's basename with
    the file extension matching the output type requested.
   * Supported keys: allowed and [default] values (file extension used by default when true)
      * `plot`: [true] or false  (.hypnodensity.png)
      * `hypnodensity`: [true] or false (.hypnodensity.txt)
      * `hypnogram`: [true] or false (.hypnogram.txt)
      * `diagnosis`: [true] or false (.diagnosis.txt)

* `filename`
   * Description
    Alternative filename to use in place of the default naming convention and path location used when saving results.  
   * Supported keys
      * `plot`: Full path for hypnodensity image file
      * `hypnodensity`: Full path for hypnodensity output text file
      * `hypnogram`: Full path for hypnogram output text file
      * `diagnosis`: Full path for diagnosis output text file

## Licensing and Attribution

This software is licensed under the Creative Commons CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/).  

Include the following citation when attributing use of this software:

Jens B. Stephansen, Alexander N. Olesen, Mads Olsen, Aditya Ambati, Eileen B. Leary, Hyatt E. Moore, Oscar Carrillo et al. ["Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy."](https://www.nature.com/articles/s41467-018-07229-3). Nature Communications 9, Article number:5229 (2018).

## Academic citation

Please use the following citation when referencing this software:

Jens B. Stephansen, Alexander N. Olesen, Mads Olsen, Aditya Ambati, Eileen B. Leary, Hyatt E. Moore, Oscar Carrillo et al. ["Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy."](https://www.nature.com/articles/s41467-018-07229-3). Nature Communications 9, Article number:5229 (2018).

```
@article{Stephansen2018,
author = {Stephansen, Jens B and Olesen, Alexander N and Olsen, Mads and Ambati, Aditya and Leary, Eileen B and Moore, Hyatt E and Carrillo, Oscar and Lin, Ling and Han, Fang and Yan, Han and Sun, Yun L and Dauvilliers, Yves and Scholz, Sabine and Barateau, Lucie and Hogl, Birgit and Stefani, Ambra and Hong, Seung Chul and Kim, Tae Won and Pizza, Fabio and Plazzi, Giuseppe and Vandi, Stefano and Antelmi, Elena and Perrin, Dimitri and Kuna, Samuel T and Schweitzer, Paula K and Kushida, Clete and Peppard, Paul E and Sorensen, Helge B. D. and Jennum, Poul and Mignot, Emmanuel},
doi = {10.1038/s41467-018-07229-3},
journal = {Nature Communications},
number = {5229},
title = {{Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy}},
volume = {9},
year = {2018}
}
```
