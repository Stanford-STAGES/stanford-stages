function c_sig = load_signal_temp(filepath,fs)

hdr = loadHDR(filepath);
ind = zeros(16,1);

for i=1:16
    try
        ind(i) = find(get_alternative_name(i,hdr.label));
    end
end


testCentral     =  (sum(ind([1 3]))==0 & (sum(ind([2 4]))==0 | sum(ind(9:10))==0));
testOccipital   =  (sum(ind([5 7]))==0 & (sum(ind([6 8]))==0 | sum(ind(9:10))==0));
testEOG         =  (sum(ind([11 13])) == 0 & (sum(ind([12 14])) == 0 | sum(ind(9:10)) == 0));
testEMG         =   ind(15) == 0 & ind(16) == 0;
if any([testCentral testOccipital testEOG testEMG])
    c_sig = [];
    hdr.label
    disp('Wrong channels')
    return
end

loaded = ind~=0;

[~,rec] = loadEDF(filepath,ind(loaded));


%% Re-reference
if ind(2)~=0
    if ind(9)~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(2)}, hdr.label{ind(9)});
        rec{2-sum(~loaded(1:2-1))} = rec{2-sum(~loaded(1:2-1))} - rec{9-sum(~loaded(1:9-1))};
    elseif ind(10) ~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(2)}, hdr.label{ind(10)});
        rec{2-sum(~loaded(1:2-1))} = rec{2-sum(~loaded(1:2-1))} - rec{10-sum(~loaded(1:10-1))};
    else
        rec(2-sum(~loaded(1:2-1))) = [];
        loaded(2) = 0;
    end
end

if ind(4)~=0
    if ind(10)~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(4)}, hdr.label{ind(10)});
        rec{4-sum(~loaded(1:4-1))} = rec{4-sum(~loaded(1:4-1))} - rec{10-sum(~loaded(1:10-1))};
    elseif ind(9) ~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(4)}, hdr.label{ind(9)});
        rec{4-sum(~loaded(1:4-1))} = rec{4-sum(~loaded(1:4-1))} - rec{9-sum(~loaded(1:9-1))};
    else
        rec(4-sum(~loaded(1:4-1))) = [];
        loaded(4) = 0;
    end
end

if ind(6)~=0
    if ind(9)~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(6)}, hdr.label{ind(9)});
        rec{6-sum(~loaded(1:6-1))} = rec{6-sum(~loaded(1:6-1))} - rec{9-sum(~loaded(1:9-1))};
    elseif ind(10) ~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(6)}, hdr.label{ind(10)});
        rec{6-sum(~loaded(1:6-1))} = rec{6-sum(~loaded(1:6-1))} - rec{10-sum(~loaded(1:10-1))};
    else
        rec(6-sum(~loaded(1:6-1))) = [];
        loaded(6) = 0;
    end
end

if ind(8)~=0
    if ind(10)~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(8)}, hdr.label{ind(10)});
        rec{8-sum(~loaded(1:8-1))} = rec{8-sum(~loaded(1:8-1))} - rec{10-sum(~loaded(1:10-1))};
    elseif ind(10) ~=0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(8)}, hdr.label{ind(10)});
        rec{8-sum(~loaded(1:8-1))} = rec{8-sum(~loaded(1:8-1))} - rec{9-sum(~loaded(1:9-1))};
    else
        rec(8-sum(~loaded(1:8-1))) = [];
        loaded(8) = 0;
    end
end

if ind(12) ~= 0
    if ind(9) ~= 0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(12)}, hdr.label{ind(9)});
        rec{12 - sum(~loaded(1:12-1))} = rec{12 - sum(~loaded(1:12-1))} - rec{9 - sum(~loaded(1:9-1))};
    elseif ind(10) ~= 0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(12)}, hdr.label{ind(10)});
        rec{12 - sum(~loaded(1:12-1))} = rec{12 - sum(~loaded(1:12-1))} - rec{10 - sum(~loaded(1:10-1))};
    end
end

if ind(14) ~= 0
    if ind(10) ~= 0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(14)}, hdr.label{ind(10)});
        rec{14 - sum(~loaded(1:14-1))} = rec{14 - sum(~loaded(1:14-1))} - rec{10 - sum(~loaded(1:10-1))};
    elseif ind(9) ~= 0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(14)}, hdr.label{ind(9)});
        rec{14 - sum(~loaded(1:14-1))} = rec{14 - sum(~loaded(1:14-1))} - rec{9 - sum(~loaded(1:9-1))};
    end
end

% EMG channel 1 ref to 2
if ind(15) ~= 0
    if ind(16) ~= 0
        fprintf('%s | Referencing %s to %s \n', datetime, hdr.label{ind(15)}, hdr.label{ind(16)});
        rec{15 - sum(~loaded(1:15-1))} = rec{15 - sum(~loaded(1:15-1))} - rec{16 - sum(~loaded(1:16 - 1))};
    end
end

if ind(9)~=0
    rec{9-sum(~loaded(1:9-1))} = [];
end
if ind(10)~=0
    rec{10-sum(~loaded(1:10-1))} = [];
end
if ind(16) ~= 0
    rec{16 - sum(~loaded(1:16 - 1))} = [];
end
rec(cellfun(@isempty,rec)) = [];
loaded([9:10 16]) = 0;
%%
sig = cell(sum(loaded),1);
cFs = hdr.fs(ind(loaded));

for i = 1:length(rec)
    idx = find(isnan(rec{i})|isinf(rec{i}));
    
    rec{i}(idx) = 0;
    SRC = dsp.SampleRateConverter('Bandwidth',50,'InputSampleRate',cFs(i),...
        'OutputSampleRate',fs,'StopbandAttenuation',30);
    sig{i} = SRC.step(rec{i});
    sig{i} = sig{i}(1:length(sig{i})-rem(length(sig{i}),fs*30));
end

noise = get_noise(sig,fs,loaded);

c_noise = Inf(16,1);
c_noise(loaded) = noise;
c_sig = cell(16,1);
c_sig(loaded) = sig;
rem1 = 1:4;
[~,eegRem1] = min(c_noise(1:4));
rem1(eegRem1) = [];
rem2 = 5:8;
[~,eegRem2] = min(c_noise(5:8));
rem2(eegRem2) = [];

c_sig([rem1,rem2]) = [];
c_sig(3:4) = []; % this removes ref channels
c_sig(cellfun(@isempty, c_sig)) = []; % this removes extra ocular channels




function synonym = get_alternative_name(ch,labels)

dict{1} = {'C3M2','C3-M2','C3-A2','C3A1','C3A2','C3M','C3-A1','C3-M1',...
    'EEG C3-A2','C3/A2','C3_A2','C3-x','C3x'};

dict{2} = {'C3','EEG C3'};
% dict{2} = {'C3'};

dict{3} = {'C4M1','C4-M1','C4-A1','C4A1','C4M','C4-A2','C4-M1','EEG C4-A1','C4/A1','C4_A1','C4x','C4-x', 'C4:A1'};

dict{4} = {'C4','EEG C4'};
% dict{4} = {'C4'};

dict{5} = {'O1M2','O1-M2','O1M2','O1-A2',...
    'O1A2','O1M','O1A2','O1AVG','O1-M2','O1-A2','O1-M1','EEG O1-A2','O1/A2','O1_A2','O1-x','O1x'};

dict{6} = {'O1','EEG O1'};
% dict{6} = {'O1'};

dict{7} = {'O2M1','O2-M1','O2M1','O2-A1',...
    'O2A1','O2M','O2A1','O2AVG','O2-M1','O2-A2','O2-M2','EEG O2-A1','O2_A1','O2-x','O2x', 'O2:A1'};

dict{8} = {'O2','EEG O2'};
% dict{8} = {'O2'};

dict{9} = {'A2','M2','EEG M2', 'EEG A2'};

dict{10} = {'A1','M1','EEG M1','A1/A2', 'EEG A1'};

dict{11} = {'LEOG','LEOG-M2','','LEOGx','LOCA2','LEOGM2','LEOC-x','LOC','LOC-A2',...
    'L-EOG','EOG-L','EOG Left','Lt.','LOC-Cz','LOCA2','LEOG-x','Lt. Eye (E1)','EOG Au-hor-L','EOG Au-hor','EOG Au-li',...
    'EOG EOG L','LOC-x','EOG_L', 'E1-M2', 'EOG LOC-A2', 'EOG1:A2'};

dict{12} = {'EOGG', 'EOG1'};

dict{13} = {'REOG','REOG-M1','REOGx','ROCA1','ROCM1','REOGM1','REOC-x','ROC-M1',...
    'ROC','ROC-A1','R-EOG','EOG-R','EOG Right','Rt.','ROC-Cz','ROCA2','REOG-x','ROC-A2','Rt. Eye (E2)','EOG Au-hor-R',...
    'Unspec Auhor','EOG Au_re','EOG EOG R','Unspec Au-hor#2','EOG Au-hor#2','ROC-x','EOG_R', 'E2-M2', 'EOG ROC-A2', 'EOG2:A2'};

dict{14} = {'EOGD', 'EOG2'};

dict{15} = {'Chin1Chin2','Chin1-Chin2','Chin EMG','Chin','CHIN EMG','Subm','Chin_L',...
    'Chin_R','CHINEMG','ChinCtr','ChinL','ChinR','Chin2EMG','ChinEMG','Chin-L',...
    'Chin-Ctr','Chin-R','EMG EMG Chin','EMG1-EMG2','EMG Ment1','EMG Ment2','EMG Ment3','Unspec subment',...
    'EMG Chin1','EMG Chin2','EMG Chin3','EMG_SM','L Chin-R Chin', 'Chin 1-Chin Z', 'Chin 2-Chin Z', 'EMG Chin', 'EMG1', 'Chin 1-Chin 2', 'EMG', 'EMG3'};

dict{16} = {'EMG2'};

for d = dict{ch}
    synonym = strcmp(d,labels);
    if sum(synonym)>0
        break
    end
end




function hjorth = extract_hjort(fs,dim,slide,input)

%Length of first dimension
dim = dim*fs;
%Specify overlap of segments in samples
slide = slide*fs;

%Creates 2D array of overlapping segments
D = buffer(input,dim,dim-slide,'nodelay');

D(:,end) = [];

%Extract Hjorth for each segment
dD = diff(D,1);
ddD = diff(dD,1);
mD2 = mean(D.^2);
mdD2 = mean(dD.^2);
mddD2 = mean(ddD.^2);

top = sqrt(mddD2 ./ mdD2);

mobility = sqrt(mdD2 ./ mD2);
activity = mD2;
complexity = top./mobility;


hjorth = [activity;complexity;mobility];

[~,b] = find(isnan(hjorth));

hjorth(:,unique(b)) = 0;

hjorth = log(hjorth+eps);

function noise = get_noise(channels,fs,loaded)
load('noiseM.mat')


noise = zeros(length(channels),1);
dist = [1 1 1 1 2 2 2 2 0 0 3 3 4 4 5 5];
dist = dist(loaded);
for i=1:length(channels)
    M = extract_hjort(fs,5*60,5*60,channels{i});
    M = num2cell(M,1);
    noise(i) = mean(cellfun(@(x) sqrt((x-noiseM.meanV{dist(i)})'*(noiseM.covM{dist(i)})^-1*(x-noiseM.meanV{dist(i)})),M));
    
end


