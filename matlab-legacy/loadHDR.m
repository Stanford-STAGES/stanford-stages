function HDR = loadHDR(fullfilename)
%HDR = loadHDR(fullfilename)
% HDR is the header of the EDF file identified by the string fullfilename


%Hyatt Moore IV (< June, 2013)

if(nargin==0)
    
    default_pathname = '.'; %uncomment this on the first run, then recomment
    [filename,pathname]=uigetfile({'*.EDF;*.edf','European Data Format'},'File Finder',...
        default_pathname);
    if(filename~=0)
        fullfilename = fullfile(pathname,filename);
    end
end
    
    
HDR = loadEDF(fullfilename);
    
    
end

