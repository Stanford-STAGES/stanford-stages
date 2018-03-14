% Author: Jens S.
% Modifier: Hyatt M.
% Modified for tutorial.

% Originally used: 
%     paths = {'/data2/psg/SSC/APOE/','/data2/psg/SSC/NARCO/',...
%         '/data2/psg/WSC_EDF/','/data2/psg/Korea/SOMNO/',...
%         '/home/jenss/triple_scored_edf/tech1/'};
%   parfor (now for)
%   sig = load_signal_temp(listF{i},channels,fs);        (load_signal_temp
%   does not take channels argument)
%   had saved noiseM.mat now saves noise.mat
%   Removed unused: 
%           dist = cell(5,1); (unused, removed)
%           pTiles = linspace(0,100,50000);     % unused (hm)
%           channels = {'C3M2','O1M2','LEOGM2','REOGM1','Chin1Chin2'}; -
%           function argmin()
% Creating input argument paths and saveDir so no longer hard coding these values.    
% Requires: 
% - load_signal_temp.m
% - extract_hjort.m

function create_noise(paths, noisePath)
    if(~iscell(paths))
        paths = {paths};
    end
    noiseFile = fullfile(noisePath,'tutorial_noiseM.mat');
    if(exist(noiseFile,'file'))
        delete(noiseFile);
    end
    fs = 100;
    
    %Filtering variables
    fcH = 0.2/(fs/2); %relative frequency/2
    
    [bH,aH] = butter(5,fcH,'high');
    
    fcL = num2cell([49 25 12.5 6.25 3.125]/(fs/2));
    
    [bL,aL] = cellfun(@(x) butter(5,x,'low'), fcL,'Un',0);
    
    wind = 60*5;
    slide = 60*2.5;
    
    %In case labels are not in the same folder as EDFs
    
    %Make list of files.
    listF = [];
    for i=1:length(paths)
        listT = dir(paths{i});
        listT = {listT(:).name};
        listT = listT(3:end);
        
        index = strfind(listT,'edf');
        index = not(cellfun('isempty',index));
        
        index2 = strfind(listT,'EDF');
        index2 = not(cellfun('isempty',index2));
        
        index = or(index,index2);
        
        listT = strcat(paths{i},listT);
        
        listF = [listF listT(index)];
    end
    
    
    cC = cell(length(listF),5);
    % cC = cell(20,5);
    % listF = listF(1:20);
    
    
    for i=1:length(listF)
        try
            fprintf('%0.2f\n',i/numel(listF));
            
            sig = load_signal_temp(listF{i},fs);
            % sig = load_signal_temp(listF{i},channels,fs);
            sig = cellfun(@(x) x(round(end/5):round(end*4/5)),sig,'UniformOutput',false);
            
            sig = cellfun(@(x) filtfilt(bH,aH,x),sig,'Un',0);
            sig = cellfun(@(x) (filtfilt(bL{1},aL{1},x)),sig,'Un',0);
            C = cell(5,1);
            for j=1:length(sig)
                C{j} = extract_hjort(fs,wind,slide,sig{j},true);
            end
            
            cC(i,:) = C;
        catch me
            showME(me);
        end
    end
    
    meanV = cell(5,1);
    covM = cell(5,1);
    for i=1:5
        concC = cC{:,i}; % = log([cC{:,i}]+eps);
        meanV{i} = mean(concC,2);
        covM{i} = cov(concC');
    end
    
    noiseM.meanV = meanV;
    noiseM.covM = covM;
    
    save(noiseFile,'noiseM')

end


