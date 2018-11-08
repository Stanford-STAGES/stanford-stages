function C = extractAC(fs,dim1,slide,input1,input2)

%Length of first dimension
dim1 = dim1*fs;
%Specify overlap of segments in samples
slide = slide*fs;

input2 = [zeros(1,dim1/2) input2 zeros(1,dim1/2)];
%Creates 2D array of overlapping segments 
D1 = buffer(input1,dim1,dim1-slide,'nodelay');
D2 = buffer(input2,dim1*2,dim1*2-slide,'nodelay');

D1 = [zeros(dim1/2,size(D1,2));D1;zeros(dim1/2,size(D1,2))];

D2 = D2(:,1:size(D1,2));
D1 = D1(:,1:size(D1,2));


%Flip data to get auto-correlation

%Fast implementation of auto/cross-correlation
C = fftshift(ifft(fft(D1,dim1*2-1,1).*conj(fft(D2,dim1*2-1,1)),[],1),1);

%Remove mirrored part
C = C(dim1/2:end-dim1/2,:);

%Scale data with log modulus
scale = log(max(abs(C)+1)/dim1);
C = bsxfun(@rdivide,C,(max(abs(C))./scale));

[~,a] = find(isnan(C));
[~,b] = find(isinf(C));

a = unique([a;b]);

C(:,a) = 0;







