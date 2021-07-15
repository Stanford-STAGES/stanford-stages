# Validating stanford-stages app's _master branch_ configuration

The sleep study CHP_040.edf is may be used to verify that you have the software setup correctly.  It can be downloaded from the following mirrors:

1. [Mirror 1](https://stanfordmedicine.box.com/shared/static/0lvvyaprzinzz7dult87t7hr96s2dnqq.edf) [380 MB]
2. [Mirror 2](https://www.informaton.org/narco/ml/validation/CHP040.edf) [380 MB]

The sleep study may be placed in any directory.  Edit the stanford_stages.json file so that the 
absolute pathname for `CHP040.edf` is listed correctly for the `edf_filename` key.  

If the .edf file was was saved to to the folder `C:/Data/ml/`, then the the json entry would be:

<pre><code>"edf_filename":  "C:/Data/ml/CHP040.edf",</code></pre>

There is a short validation check which uses one model and a full validation check that uses all 16 models.
We recommend starting with the short validation check first in order to more quickly troubleshoot any issues or differences that may come up, before moving on to the full validation check which takes longer to complete.

## Quick validation check (one model)

This validation tests your configuration using one model, 'ac_rh_ls_lstm_01'.  

1. Edit the inf_config.py file so that the model_used property is set as follows:

   <pre><code>self.models_used = ['ac_rh_ls_lstm_01']</code></pre>

2. Run the shell script from a command line terminal as:

   <pre><code>sh ./verify_chp_040.sh</code></pre>

3. Check results

   Upon successful completion, a hypnogram file and hypnodensity image will be created
and saved in the same directory as the input CHP040.edf file.  

   Expected results for the ac_rh_ls_lstm_01 hypnogram and hypnodensity results can be found here:

    * [Hypnogram (single model)](https://github.com/Stanford-STAGES/stanford-stages/blob/master/documentation/validation_files/master_branch/single_model/CHP040.hypnogram.txt) [4 KB]

    * [Hypnodensity image (single model)](https://github.com/Stanford-STAGES/stanford-stages/blob/master/documentation/validation_files/master_branch/single_model/CHP040.hypnodensity.png) [158 KB]

   Expected diagnosis output is:

   <pre>Score: -0.0076
   Diagnosis: Narcolepsy type 1</pre>

## Full validation check (all models)

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

2. Run the shell or batch script from a command line terminal or prompt, depending on your operating system.

   * Windows PC
     <pre><code>sh ./verify_chp_040.bat</code></pre>

   * Mac OSX or Linux
     <pre><code>sh ./verify_chp_040.sh</code></pre>

   Note: If the shell script has been run before using a single model, you will need to delete the previously cached "pickle" files.
This requires editing the shell script and uncommenting the line that removes saved pickle files. For Mac OSX and Linux users, edit the verify_chp_040.sh script and uncomment (remove the '#') the 
line <pre><code># rm /Users/unknown/data/sleep/narcoTest/CHP040.\*pkl</code></pre> in order 
to remove any previously saved pickle files from the output directory.

3. Check results

   Upon successful completion, a hypnogram file and hypnodensity image will be created
and saved in the same directory as the input CHP040.edf file.  

   Expected results for may be found here:

    * [Hypnogram (full)](https://github.com/Stanford-STAGES/stanford-stages/blob/master/documentation/validation_files/master_branch/all_models/CHP040.hypnogram.txt) [4 KB]

    * [Hypnodensity image (full)](https://github.com/Stanford-STAGES/stanford-stages/blob/master/documentation/validation_files/master_branch/all_models/CHP040.hypnodensity.png) [155 KB]

   Expected diagnosis output is:

   <pre>Score: 0.1658
   Diagnosis: Narcolepsy type 1</pre>
