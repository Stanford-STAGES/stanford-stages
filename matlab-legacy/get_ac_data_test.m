function get_ac_data_test



%This script generates the auto/cross-correlation data used for the convolutional neural networks.

%Initialize variables

fs = 100;

%Filtering variables
fcH = 0.2/(fs/2); %relative frequency/2

[bH,aH] = butter(5,fcH,'high');

fcL = num2cell([49 25 12.5 6.25 3.125]/(fs/2));

[bL,aL] = cellfun(@(x) butter(5,x,'low'), fcL,'Un',0);

segLength = 120*10;
overlap = 40*10;
% addpath('/home/jenss/Documents/MATLAB/13_10_15/sev')
% import filter.*
% import plist.*;

%In case labels are not in the same folder as EDFs

saveDir = '/home/neergaard/tmp/chp040/';
% saveDir = '/home/neergaard/StanfordProfMignot/h5/';
% saveDir = '/media/neergaard/neergaardhd/missing_data/h5/';
% saveDir = '/home/jenss/ac_data_test/';
%paths = {'/home/jenss/triple_scored_edf/tech1/'}; %,'/home/jenss/June2015/','/data2/psg/SSC/NARCO/',...
%	'/data2/psg/Korea/SOMNO/'};
%paths = {'/data2/psg/Innsbruck/','/data2/psg/SSC/NARCO/','/data2/psg/Korea/SOMNO/'};
%paths = {'/data2/psg/SSC/APOE/','/data2/psg/WSC_EDF/'};
%paths = {'/home/jenss/June2015/'};
%pathsL = '/home/jenss/italy_data/';
%paths = dir(pathsL); paths = paths(3:end);
%pathFlag = [paths.isdir];
%paths = {paths(pathFlag).name}
%paths = strcat(pathsL,paths,'/')

% paths = {'/data2/psg/Jazz/SXB15/'}
% paths = {'/media/neergaard/neergaardhd/missing_data/edf/'};
% paths = {'/home/neergaard/StanfordProfMignot/PSG/'};
paths = {saveDir};

%Make list of files.
listF = [];
paths = sort(paths);


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
    
%     rng(12345)
    listT = listT(index);
%     listT = listT(randperm(length(listT)));
    %     listT = listT(1:300);
    listF = [listF listT];
end

% listF = strcat(paths{i},...
% {'JAZZ15-006_400_001.EDF',...
% 'JAZZ15-009_401_001.EDF',...
% 'JAZZ15-013_406_001.EDF',...
% 'JAZZ15-013_408_001.EDF',...
% 'JAZZ15-017_414_001.EDF',...
% 'JAZZ15-018_406_001.EDF',...
% 'JAZZ15-041_428_001.EDF',...
% 'JAZZ15-041_433_001.EDF',...
% 'JAZZ15-046_407_001.EDF',...
% 'JAZZ15-050_403_001.EDF',...
% 'JAZZ15-053_419_001.EDF',...
% 'JAZZ15-087_401_001.EDF',...
% 'JAZZ15-087_407_001.EDF',...
% 'JAZZ15-088_400_001.EDF',...
% 'JAZZ15-088_402_002.EDF',...
% 'JAZZ15-088_408_002.EDF',...
% 'JAZZ15-088_410_001.EDF',...
% 'JAZZ15-089_401_001.EDF'});


%rng(12345)
% rng('shuffle')
% listF = listF(randperm(length(listF)));

% channels = {'C3M2','O1M2','LEOGM2','REOGM1','Chin1Chin2','EKG'};

%[v1,v2,v3] = hypstartstop;

for i=1:length(listF)
    tic
    try
        disp(num2str(i))
        dataName = listF{i}(1:end-4);
        ind = strfind(dataName,'/');
        name = dataName(ind(end)+1:end);
        dataName = ['data_' name];
        
        if exist([saveDir dataName '.h5'],'file')
            continue
        end
        disp([saveDir dataName])
        
        
        sig = load_signal_temp(listF{i},fs);
        
        if isempty(sig)
            continue
        end
        
        hyp = load_scored_dat([listF{i}(1:end-4) '.STA']);
        
        if isempty(hyp)
            hyp = ones(length(sig{1})/(fs*30),1);
            %hyp(1:v2(i)) = 7;
            %continue
        end
        
        
        hyp = hyp(1:length(sig{1})/(fs*30));
        label = repmat(hyp',120,1);
        label = label(:);
        
        sig = cellfun(@(x) filtfilt(bH,aH,x),sig,'Un',0);
        sigH = sig;
        sig = cellfun(@(x) single(filtfilt(bL{1},aL{1},x)),sig,'Un',0);
        C = cell(6,1);
        disp('Getting autocorr')
        for j=1:length(sig)
            switch j
                case 1
                    dim = 2;
                case 3
                    dim = 4;
                case 5
                    dim = 0.4;
            end
            C{j} = extractAC(fs,dim,0.25,sig{j}',sig{j}');
        end
        C{6} = extractAC(fs,4,0.25,sig{3}',sig{4}');
        disp('Autocorr complete')
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
        
        save_data(saveDir,dataName,C,labels)
    end
    toc
end

function save_data(saveDir,dataName,C,labels)

disp(['Saving ..' dataName])
here = cd;
if ~exist(saveDir, 'dir')
    mkdir(saveDir)
end
cd(saveDir)

h5create([dataName '.h5'],'/data',size(C));
h5write([dataName '.h5'], '/data', C);

h5create([dataName '.h5'],'/labels', size(labels));
h5write([dataName '.h5'], '/labels', labels);

cd(here)





