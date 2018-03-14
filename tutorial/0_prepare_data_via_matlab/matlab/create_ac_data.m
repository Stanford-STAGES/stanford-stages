% Author: Jens S.
% Modifier: Hyatt M.

%This script generates the auto/cross-correlation data used for the convolutional neural networks.

% Requires:
% 1. load_signal_temp.m
% 2. load_scored_data.m

% Modifications:
% 1. made into a function that takes arguments for load and save pathnames
% 2. removed try parpool(4) end;
% 3. parfor to for (parfor j=1:5%length(sig))
% 4. cd -> pwd;
% 5. i<length(listF) -> i<=length(listF)  % go ahead and use all the data
function create_ac_data(paths, saveDir)
    if(~iscell(paths))
        paths = {paths};
    end
    
    %Initialize variables
    
    fs = 100;
    
    %Filtering variables
    fcH = 0.2/(fs/2); %relative frequency/2
    
    [bH,aH] = butter(5,fcH,'high');
    
    fcL = num2cell([49 25 12.5 6.25 3.125]/(fs/2));
    
    [bL,aL] = cellfun(@(x) butter(5,x,'low'), fcL,'Un',0);
    
    segLength = 120*10;
    overlap = 40*10;

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
        
        rng(12345)
        listT = listT(index);
        listT = listT(randperm(length(listT)));
        try
            listT(1:300) = [];
        catch me
            showME(me);
        end
        listF = [listF listT];
    end

    rng(12345)
    listF = listF(randperm(length(listF)));

    i = 0;
    while i<numel(listF)
        i = i+1;
        try
            if i<=length(listF)  % go ahead and use all the data
                dataName = listF{i}(1:end-4);
                fprintf('%s (%0.2f)\n',dataName,i/numel(listF));
                
                % ind = strfind(dataName,'/');
                % name = dataName(ind(end)+1:end);
                % dataName = ['data_' name];
                
                hyp = load_scored_data([listF{i}(1:end-4) '.STA']);
                if isempty(hyp)
                    disp('Skipping due to missing Hyp')
                    continue
                end
                
                sig = load_signal_temp(listF{i},fs);
                if isempty(sig)
                    disp('Skipping due to missing sig')
                    continue
                end
                
                hyp = hyp(1:length(sig{1})/(fs*30));
                label = repmat(hyp',120,1);
                label = label(:);
                
                sig = cellfun(@(x) filtfilt(bH,aH,x),sig,'Un',0);
                sig = cellfun(@(x) single(filtfilt(bL{1},aL{1},x)),sig,'Un',0);
                C = cell(6,1);
                dim = [2 2 4 4 0.4 4];
                %                 parfor j=1:5%length(sig)
                for j=1:5%length(sig)
                    
                    C{j} = extractAC(fs,dim(j),0.25,sig{j}',sig{j}');
                end
                C{6} = extractAC(fs,dim(6),0.25,sig{3}',sig{4}');
                
                disp('Data extracted')
                
                C = cellfun(@(x) x(:,1:size(C{6},2)),C,'Un',0);
                C = vertcat(C{1},C{2},C{3},C{4},C{6},C{5});
                label = label(1:size(C,2));
                C(:,label==7) = [];
                label(label==7) = [];
                hyp(hyp==7) = [];
                
                labels = zeros(5,length(label));
                
                
                for j=1:5
                    labels(j,label==j) = 1;
                end
                
                index = num2cell(buffer(1:size(C,2),segLength,overlap,'nodelay'),1);
                index(end) = [];
                
                M = cellfun(@(x) C(:,x),index,'Un',0);
                M = cat(3,M{:});
                
                L = cellfun(@(x) labels(:,x),index,'Un',0);
                L = cat(3,L{:});
                
                hyp = repmat(hyp,1,2)';
                hyp = hyp(:);
                mask = zeros(size(hyp));
                dhyp = double(abs(sign(diff(hyp))));
                mask(1:length(dhyp)) = mask(1:length(dhyp)) + dhyp;
                mask(2:length(dhyp)+1) = mask(2:length(dhyp)+1) + dhyp;
                mask = single(not(mask));
                
                %Adjust weights depending on stage.
                weight = zeros(size(hyp));
                weight(hyp==1) = 1.5;
                weight(hyp==2) = 2;
                weight(hyp==3) = 1;
                weight(hyp==4) = 2.5;
                weight(hyp==5) = 2;
                
                weight = weight.*mask;
                
                weight = repmat(weight',60,1);
                weight = weight(:);
                
                W = cellfun(@(x) weight(x)',index,'Un',0);
                W = cat(1,W{:})';
                
                if ~exist('dataStack','var')
                    dataStack = (M);
                    labelStack = (L);
                    weightStack = (W);
                else
                    dataStack = cat(3,dataStack,M);
                    labelStack = cat(3,labelStack,L);
                    weightStack = cat(2,weightStack,W);
                end
                
            end
        catch me
            showME(me);
            warning([listF{i} ' caused an error'])
        end
        if size(dataStack,3) > 900 || (i==length(listF) && size(dataStack,3) > 300)
            
            if ~exist(saveDir,'dir')
                mkdir(saveDir);
            end
            curDir = pwd;
            cd(saveDir);
            
            
            rng('shuffle');
            
            ind = randperm(size(dataStack,3));
            dataStack = dataStack(:,:,ind);
            labelStack = labelStack(:,:,ind);
            weightStack = weightStack(:,ind);
            saveName = [num2str(randi([10000000 19999999])) '.h5'];
            
            h5create(saveName,'/trainD',[1640 segLength 270]);
            h5write(saveName, '/trainD', dataStack(:,:,1:270));
            
            h5create(saveName,'/trainL',[5 segLength 270]);
            h5write(saveName, '/trainL', labelStack(:,:,1:270));
            
            h5create(saveName,'/trainW',[segLength 270]);
            h5write(saveName, '/trainW', weightStack(:,1:270));
            
            h5create(saveName,'/keep',[segLength 270]);
            h5write(saveName, '/keep', ones(segLength,270));
            
            h5create(saveName,'/valD',[1640 segLength 30]);
            h5write(saveName, '/valD', dataStack(:,:,271:300));
            
            h5create(saveName,'/valL',[5 segLength 30]);
            h5write(saveName, '/valL', labelStack(:,:,271:300));
            
            
            dataStack(:,:,1:300) = [];
            labelStack(:,:,1:300) = [];
            weightStack(:,1:300) = [];
            
            cd(curDir);
        end
    end
end

