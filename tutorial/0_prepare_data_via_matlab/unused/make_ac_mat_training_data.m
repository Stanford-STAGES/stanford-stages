addpath ../matlab;
edfPath = fullfile(pwd,'../data/edf_sleep_stage_training/');
saveDir = fullfile(pwd,'../data/ac_mat_sleep_stage_training/');
create_autocorr_data(edfPath,saveDir);