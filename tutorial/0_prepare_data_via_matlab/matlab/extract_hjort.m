function hjorth = extract_hjort(fs,dim,slide,input,remNaN)
if(nargin<5)
    remNaN = false;
end
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
    
    if remNaN
        hjorth(:,unique(b)) = [];
    else
        hjorth(:,unique(b)) = 0;
    end
    
    hjorth = log(hjorth+eps);
end
