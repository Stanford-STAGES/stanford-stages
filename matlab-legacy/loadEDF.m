function [HDR, signal] = loadEDF(filename,channels)
%[HDR, signal] = loadEDF(filename,channels)
%Loads EDF files (not EDF+ format), if only one output is specified then
%only the header information (HDR) is loaded.
%channels is a vector of the numeric signals to be loaded.  If left blank,
%then all of the channels in the EDF will be loaded.  
%Written by Hyatt Moore
%last modified: October, 8, 2012
%July 13, 2013: fixed a bug which could cause incorrect visual scaling in
%cases where the channel units of measurement did not match up with the
%index k.  

if(nargin==0)
    disp 'No input filename given; aborting';
    return;
end;
if(nargin==1)
    channels = 0;
end;
if(nargin>2)
    disp('Too many input arguments in loadEDF.  Extra input arguments are ignored');
end;

%handle filenames with unicode characters in them
filename = char(unicode2native(filename,'utf-8'));
fid = fopen(filename,'r');
precision = 'uint8';

HDR.ver = str2double(char(fread(fid,8,precision)'));% 8 ascii : version of this data format (0) 
HDR.patient = char(fread(fid,80,precision)');% 80 ascii : local patient identification (mind item 3 of the additional EDF+ specs)')
HDR.local = char(fread(fid,80,precision)');% 80 ascii : local recording identification (mind item 4 of the additional EDF+ specs)')
HDR.startdate = char(fread(fid,8,precision)');% 8 ascii : startdate of recording (dd.mm.yy)') (mind item 2 of the additional EDF+ specs)')
HDR.starttime = char(fread(fid,8,precision)');% 8 ascii : starttime of recording (hh.mm.ss)') 
HDR.HDR_size_in_bytes = str2double(char(fread(fid,8,precision)'));% 8 ascii : number of bytes in header record 
HDR.reserved = char(fread(fid,44,precision)');% 44 ascii : reserved 
HDR.number_of_data_records = str2double(char(fread(fid,8,precision)'));% 8 ascii : number of data records (-1 if unknown, obey item 10 of the additional EDF+ specs)')  %236
HDR.duration_of_data_record_in_seconds = str2double(char(fread(fid,8,precision)'));% 8 ascii : duration of a data record, in seconds 
HDR.num_signals = str2double(char(fread(fid,4,precision)'));% 4 ascii : number of signals (ns)') in data record 
ns = HDR.num_signals;

datetime = [HDR.startdate, '.' , HDR.starttime];
HDR.T0 = zeros(1,6); %[year(4) month(2) day(2) hour(2) minute(2) second(2)]
try
    for k=1:6
        [str, datetime] = strtok(datetime,'.');
        HDR.T0(k) = str2num(str);
    end
    yy = HDR.T0(3);
    dd = HDR.T0(1);
    HDR.T0(3) = dd;
    if(yy>=85)
        yy = yy+1900;
    else
        yy = yy+2000;
    end;
    HDR.T0(1) = yy;
catch ME
    disp(['Failed to load the date/time in this EDF.  Filename: ', filename]);
end

%ns = number of channels/signals in the EDF
%duration_of_signal_in_samples = 

HDR.label = cellstr(char(fread(fid,[16,ns],precision)'));% ns * 16 ascii : ns * label (e.g. EEG Fpz-Cz or Body temp)') (mind item 9 of the additional EDF+ specs)')
HDR.transducer = cellstr(char(fread(fid,[80,ns],precision)'));% ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)')
HDR.physical_dimension = cellstr(char(fread(fid,[8,ns],precision)'));% ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)')
HDR.physical_minimum = str2double(cellstr(char(fread(fid,[8,ns],precision)')));% ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)')
HDR.physical_maximum = str2double(cellstr(char(fread(fid,[8,ns],precision)')));% ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)')
HDR.digital_minimum = str2double(cellstr(char(fread(fid,[8,ns],precision)')));% ns * 8 ascii : ns * digital minimum (e.g. -2048)')
HDR.digital_maximum = str2double(cellstr(char(fread(fid,[8,ns],precision)')));% ns * 8 ascii : ns * digital maximum (e.g. 2047)')
HDR.prefiltering = cellstr(char(fread(fid,[80,ns],precision)'));% ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)')
HDR.number_samples_in_each_data_record = str2double(cellstr(char(fread(fid,[8,ns],precision)')));% ns * 8 ascii : ns * nr of samples in each data record
HDR.reserved = cellstr(char(fread(fid,[32,ns],precision)'));% ns * 32 ascii : ns * reserved

HDR.fs = HDR.number_samples_in_each_data_record/HDR.duration_of_data_record_in_seconds; %sample rate
HDR.samplerate = HDR.fs;
HDR.duration_sec = HDR.duration_of_data_record_in_seconds*HDR.number_of_data_records;
HDR.duration_samples = HDR.duration_sec*HDR.fs;

if(nargout>1)

    if(channels == 0) %requesting all channels then
        channels = 1:HDR.num_signals;
    end;

    signal = cell(numel(channels),1);
    bytes_per_sample = 2;
    for k = 1:numel(channels)        
        cur_channel = channels(k);
        if(cur_channel>0 && cur_channel<=HDR.num_signals)
            physical_dimension = HDR.physical_dimension{cur_channel};% ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)')
            if(strcmpi(physical_dimension,'mv'))
                scale = 1e3;
            elseif(strcmpi(physical_dimension,'v'))
                scale = 1e6;
            else
                scale = 1;
            end

            num_samples_in_cur_data_record = HDR.number_samples_in_each_data_record(cur_channel);
            precision = [num2str(num_samples_in_cur_data_record),'*int16'];
            skip = (sum(HDR.number_samples_in_each_data_record)-num_samples_in_cur_data_record)*bytes_per_sample; %*2 because there are two bytes used for each integer

            cur_channel_offset = sum(HDR.number_samples_in_each_data_record(1:cur_channel-1))*bytes_per_sample; %*2 because there are two bytes used for each integer
            offset = HDR.HDR_size_in_bytes+cur_channel_offset;
            fseek(fid,offset,'bof');
            signal{k} = fread(fid,HDR.duration_samples(cur_channel),precision,skip);
            signal{k} = scale*(HDR.physical_minimum(cur_channel)+(signal{k}(:)-HDR.digital_minimum(cur_channel))...
                *(HDR.physical_maximum(cur_channel)-HDR.physical_minimum(cur_channel))...
                /(HDR.digital_maximum(cur_channel)-HDR.digital_minimum(cur_channel)));
        end;
    end;
end;

% The voltage (i.e. signal) in the file by definition equals
% [(physical miniumum)
% + (digital value in the data record - digital minimum) 
% x (physical maximum - physical minimum) 
% / (digital maximum - digital minimum)].
fclose(fid);

% HEADER Specs...
% 8 ascii : version of this data format (0)') 
% 80 ascii : local patient identification (mind item 3 of the additional EDF+ specs)')
% 80 ascii : local recording identification (mind item 4 of the additional EDF+ specs)')
% 8 ascii : startdate of recording (dd.mm.yy)') (mind item 2 of the additional EDF+ specs)')
% 8 ascii : starttime of recording (hh.mm.ss)') 
% 8 ascii : number of bytes in header record 
% 44 ascii : reserved 
% 8 ascii : number of data records (-1 if unknown, obey item 10 of the additional EDF+ specs)') 
% 8 ascii : duration of a data record, in seconds 
% 4 ascii : number of signals (ns)') in data record 
% ns * 16 ascii : ns * label (e.g. EEG Fpz-Cz or Body temp)') (mind item 9 of the additional EDF+ specs)')
% ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)') 
% ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)') 
% ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)') 
% ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)') 
% ns * 8 ascii : ns * digital minimum (e.g. -2048)') 
% ns * 8 ascii : ns * digital maximum (e.g. 2047)') 
% ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)') 
% ns * 8 ascii : ns * nr of samples in each data record 
% ns * 32 ascii : ns * reserved



%ALL of this can be improved to all for a directory name to be passed
%the function will need to check if the last char is a '/' or not though
%which is simple enough using the 'end' parameter of the matrix
% DIRECTORY_NAME = 'EE Training Set'; %this is relative to the directory
% DIRECTORY_NAME = '/Users/hyatt4/Documents/Sleep Project/EE Training Set/'; %this is absolute
% DIRECTORY_NAME = '/Users/hyatt4/Documents/Sleep Project/EE Training Set/'; %this is absolute
% file_list = dir([DIRECTORY_NAME '*.EDF'])');  
% 
% for i = 1:length(file_list)')
%     if(~file_list(i)').isdir)')
%         file_name = [DIRECTORY_NAME file_list(i)').name]
%     end;
% end;

%[s,HDR]=sload('EE Training Set/A0097_4 174733.EDF')'); %107 seconds..
% [s,HDR] = loadEDF(filename); 5.9 seconds
%[s,HDR]=sload('EE Training Set/A0097_4 174733.EDF','r',3)'); %26.2%seconds..