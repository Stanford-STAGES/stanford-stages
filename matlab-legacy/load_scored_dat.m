function hyp = load_scored_data(pathname)

fID = fopen(pathname);  % open file
if fID==-1
    fID = fopen([pathname(end-2:end) 'sta']);
end
if fID==-1
    hyp = [];
    return
end

hyp = cell2mat(textscan(fID, '%*u%u%*[^\n]'));

hyp2 = zeros(size(hyp));
hyp2(hyp==0) = 1;
hyp2(hyp==1) = 2;
hyp2(hyp==2) = 3;
hyp2(hyp==3) = 4;
hyp2(hyp==4) = 4;
hyp2(hyp==5) = 5;
hyp2(hyp2==0) = 7;

unique(hyp2);

fclose(fID);  % close file
hyp = [hyp2;ones(2000,1)*7];

