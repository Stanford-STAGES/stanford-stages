addpath matlab/;
curDir = fileparts(mfilename('fullpath'));
edfPath = fullfile(curDir,'../data/edf_sleep_stage_training/');
saveDir = fullfile(curDir,'../data/ac_h5_sleep_stage_training/');
create_ac_data(edfPath,saveDir);