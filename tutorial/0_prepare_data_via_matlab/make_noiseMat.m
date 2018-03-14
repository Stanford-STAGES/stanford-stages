addpath matlab/;
curDir = fileparts(mfilename('fullpath'));
edfPath = fullfile(curDir,'../data/edf_sleep_stage_training/');
noisePath = fullfile(curDir,'../data/');
create_noise(edfPath,noisePath);