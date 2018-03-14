% Author: Jens S.
% Modifier: Hyatt M.

% Originally from quality_control.m
% modifications include:
%     paths = {'/data2/psg/SSC/APOE/','/data2/psg/SSC/NARCO/',...
%         '/data2/psg/WSC_EDF/','/data2/psg/Korea/SOMNO/',...
%         '/home/jenss/triple_scored_edf/tech1/'};
%   parfor (now for)
%   sig = load_signal_temp(listF{i},channels,fs);        (load_signal_temp
%   does not take channels argument)
%   had saved noiseM.mat now saves noise.mat
%   Cleaned up save_data method; more robust now for getting filename saved
%   regardless of filename path syntax 
%   Removed unused: 
%           dist = cell(5,1); (unused, removed)
%           pTiles = linspace(0,100,50000);     % unused (hm)
%           channels = {'C3M2','O1M2','LEOGM2','REOGM1','Chin1Chin2'}; -
%           function argmin()
% Creating input argument paths and saveDir so no longer hard coding these values.    
% Requires: 
% - load_signal_temp.m
% - extract_hjort.m

function create_autocorr_data(paths, saveDir)
    if(~iscell(paths))
        paths = {paths};
    end
    curPath = fileparts(mfilename('fullpath'));
    noiseFile = fullfile(curPath,'noise.mat');
    
    if ~exist(noiseFile,'file')
        fprintf('Noise file not found: %s\n\tNothing done.',noiseFile);
        throw(MException('INF:File:NotFound','noise.mat file not found'));
    end
    
    fs = 100;
    
    %Filtering variables
    fcH = 0.2/(fs/2); %relative frequency/2
    
    [bH,aH] = butter(5,fcH,'high');
    
    fcL = num2cell([49 25 12.5 6.25 3.125]/(fs/2));
    
    [bL,aL] = cellfun(@(x) butter(5,x,'low'), fcL,'Un',0);
    
    wind = 60*5;
    
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
    load(noiseFile);
    
    
    for i=1:length(listF)
        try
            fprintf('%0.2f\n',i/numel(listF));
            
            % sig = load_signal_temp(listF{i},channels,fs);            
            sig = load_signal_temp(listF{i},fs);            
            
            sig = cellfun(@(x) filtfilt(bH,aH,x),sig,'Un',0);
            sig = cellfun(@(x) (filtfilt(bL{1},aL{1},x)),sig,'Un',0);
            C = cell(5,1);
            for j=1:length(sig)
                C{j} = extract_hjort(fs,wind,wind,sig{j},false);
            end
            
            cC(i,:) = C;
        catch me
            showME(me);            
        end
    end
    
    for i=1:length(listF)
        D = cell(5,1);
        
        for j=1:5
            M = num2cell(log(cC{i,j}),1);            
            D{j} = cellfun(@(x) sqrt((x-noise.meanV{j})'*(noise.covM{j})^-1*(x-noise.meanV{j})),M);
        end
        
        save_data(saveDir,listF{i},D)
    end
    
end
function save_data(saveDir,filename,data)
    [~,filename,~] = fileparts(filename);
    
    if ~exist(saveDir,'dir')
        mkdir(saveDir)
    end
    
    save(fullfile(saveDir,[filename '.mat']),'data')
end


