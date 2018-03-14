% Modifier: Hyatt
% Added 'L Chin-R Chin' to chin list
% changed 'noiseM' references to use get_noise

% Requires
% 1. get_noise.m
% 2. showME.m
function c_sig = load_signal_temp(filepath,fs)

    hdr = loadHDR(filepath);
    ind = zeros(13,1);

    for i=1:13
        try
            matchVec = get_alternative_name(i,hdr.label);
            if(any(matchVec))
                ind(i) = find(matchVec);
            end
        catch me
            showME(me);
        end
    end
    
    test1 =  (sum(ind([1 3]))==0 & (sum(ind([2 4]))==0 | sum(ind(9:10))==0));
    test2 =  (sum(ind([5 7]))==0 & (sum(ind([6 8]))==0 | sum(ind(9:10))==0));
    test3 = prod(ind(11:13))==0;
    if any([test1 test2 test3])
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
            rec{2-sum(~loaded(1:2-1))} = rec{2-sum(~loaded(1:2-1))} - rec{9-sum(~loaded(1:9-1))};
        elseif ind(10) ~=0
            rec{2-sum(~loaded(1:2-1))} = rec{2-sum(~loaded(1:2-1))} - rec{10-sum(~loaded(1:10-1))};
        else
            rec(2-sum(~loaded(1:2-1))) = [];
            loaded(2) = 0;
        end
    end
    
    if ind(4)~=0
        if ind(10)~=0
            rec{4-sum(~loaded(1:4-1))} = rec{4-sum(~loaded(1:4-1))} - rec{10-sum(~loaded(1:10-1))};
        elseif ind(9) ~=0
            rec{4-sum(~loaded(1:4-1))} = rec{4-sum(~loaded(1:4-1))} - rec{9-sum(~loaded(1:9-1))};
        else
            rec(4-sum(~loaded(1:4-1))) = [];
            loaded(4) = 0;
        end
    end
    
    if ind(6)~=0
        if ind(9)~=0
            rec{6-sum(~loaded(1:6-1))} = rec{6-sum(~loaded(1:6-1))} - rec{9-sum(~loaded(1:9-1))};
        elseif ind(10) ~=0
            rec{6-sum(~loaded(1:6-1))} = rec{6-sum(~loaded(1:6-1))} - rec{10-sum(~loaded(1:10-1))};
        else
            rec(6-sum(~loaded(1:6-1))) = [];
            loaded(6) = 0;
        end
    end
    
    if ind(8)~=0
        if ind(10)~=0
            rec{8-sum(~loaded(1:8-1))} = rec{8-sum(~loaded(1:8-1))} - rec{10-sum(~loaded(1:10-1))};
        elseif ind(10) ~=0
            rec{8-sum(~loaded(1:8-1))} = rec{8-sum(~loaded(1:8-1))} - rec{9-sum(~loaded(1:9-1))};
        else
            rec(8-sum(~loaded(1:8-1))) = [];
            loaded(8) = 0;
        end
    end
    
    if ind(9)~=0
       rec{9-sum(~loaded(1:9-1))} = [];
    end
    if ind(10)~=0
       rec{10-sum(~loaded(1:10-1))} = [];
    end    
    rec(cellfun(@isempty,rec)) = [];
    loaded(9:10) = 0;
    %%
    sig = cell(sum(loaded),1);
    cFs = hdr.fs(ind(loaded));
    
    for i = 1:length(rec)
        %         ind = find(isnan(rec{i})|isinf(rec{i}));
        %         rec{i}(ind) = 0;
        rec{i}(isnan(rec{i})|isinf(rec{i})) = 0;
        
        SRC = dsp.SampleRateConverter('Bandwidth',50,'InputSampleRate',cFs(i),...
                'OutputSampleRate',fs,'StopbandAttenuation',30);
        sig{i} = SRC.step(rec{i});
        sig{i} = sig{i}(1:length(sig{i})-rem(length(sig{i}),fs*30));
    end
    
    noise = get_noise(sig,fs,loaded);
    if(isempty(noise))
        c_sig = sig;
    else
        c_noise = Inf(13,1);
        c_noise(loaded) = noise;
        c_sig = cell(13,1);
        c_sig(loaded) = sig;
        rem1 = 1:4;
        [~,eegRem1] = min(c_noise(1:4));
        rem1(eegRem1) = [];
        rem2 = 5:8;
        [~,eegRem2] = min(c_noise(5:8));
        rem2(eegRem2) = [];
        
        c_sig([rem1,rem2]) = [];
        c_sig(3:4) = [];
    end
	
end

function synonym = get_alternative_name(ch,labelToMatch)

    dict{1} = {'C3M2','C3-M2','C3-A2','C3A1','C3A2','C3M','C3-A1','C3-M1',...
        'EEG C3-A2','C3/A2','C3_A2','C3-x','C3x','C3'};
    
    dict{2} = {'C3','EEG C3'};
    
    dict{3} = {'C4M1','C4-M1','C4-A1','C4A1','C4M','C4-A2','C4-M1','EEG C4-A1','C4/A1','C4_A1','C4x','C4-x','C4'};
    
    dict{4} = {'C4','EEG C4'};
    
    dict{5} = {'O1M2','O1-M2','O1M2','O1-A2',...
        'O1A2','O1M','O1A2','O1AVG','O1-M2','O1-A2','O1-M1','EEG O1-A2','O1/A2','O1_A2','O1-x','O1x','O1'};
    
    dict{6} = {'O1','EEG O1'};
    
    dict{7} = {'O2M1','O2-M1','O2M1','O2-A1',...
        'O2A1','O2M','O2A1','O2AVG','O2-M1','O2-A2','O2-M2','EEG O2-A1','O2_A1','O2-x','O2x','O2'};
    
    dict{8} = {'O2','EEG O2'};
    
    dict{9} = {'A2','M2','EEG M2'};
    
    dict{10} = {'A1','M1','EEG M1','A1/A2'};
    
    dict{11} = {'LEOG','LEOG-M2','','LEOGx','LOCA2','LEOGM2','LEOC-x','LOC','LOC-A2',...
        'L-EOG','EOG-L','EOG Left','Lt.','LOC-Cz','LOCA2','LEOG-x','Lt. Eye (E1)','EOG Au-hor-L','EOG Au-hor','EOG Au-li',...
        'EOG EOG L','LOC-x','EOG_L'};
    
    dict{12} = {'REOG','REOG-M1','REOGx','ROCA1','ROCM1','REOGM1','REOC-x','ROC-M1',...
        'ROC','ROC-A1','R-EOG','EOG-R','EOG Right','Rt.','ROC-Cz','ROCA2','REOG-x','ROC-A2','Rt. Eye (E2)','EOG Au-hor-R',...
        'Unspec Auhor','EOG Au_re','EOG EOG R','Unspec Au-hor#2','EOG Au-hor#2','ROC-x','EOG_R'};
    
    dict{13} = {'Chin1Chin2','Chin1-Chin2','Chin EMG','Chin','CHIN EMG','Subm','Chin_L',...
        'Chin_R','CHINEMG','ChinCtr','ChinL','ChinR','Chin2EMG','ChinEMG','Chin-L',...
        'Chin-Ctr','Chin-R','EMG EMG Chin','EMG1-EMG2','EMG Ment1','EMG Ment2','EMG Ment3','Unspec subment',...
        'EMG Chin1','EMG Chin2','EMG Chin3','EMG_SM','L Chin-R Chin'};
    
    for d = dict{ch}
        synonym = strcmp(d,labelToMatch);
        if sum(synonym)>0
            break
        end
    end
    
    
    
end
