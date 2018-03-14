% modified: Check for noise.mat in same directory as get_noise function
% (i.e. this file).  Throw error if the file is not found.
function noise = get_noise(channels,fs,loaded)
    curPath = fileparts(mfilename('fullpath'));
    noiseFile = fullfile(curPath,'../../data/','tutorial_noiseM.mat');
    if(~exist(noiseFile,'file'))
        noise = [];
        %throw(MException('INF:File:NotFound','noise.mat file not found'));
    else
        try
            noiseStruct = load(noiseFile);            
            noise = zeros(length(channels),1);
            dist = [1 1 1 1 2 2 2 2 0 0 3 4 5];
            dist = dist(loaded);
            for i=1:length(channels)
                M = extract_hjort(fs,5*60,5*60,channels{i});
                M = num2cell(M,1);
                noise(i) = mean(cellfun(@(x) sqrt((x-noiseStruct.noiseM.meanV{dist(i)})'*(noiseStruct.noiseM.covM{dist(i)})^-1*(x-noiseStruct.noiseM.meanV{dist(i)})),M));
            end
        catch me
            showME(me);
            noise = [];
        end
    end
end
