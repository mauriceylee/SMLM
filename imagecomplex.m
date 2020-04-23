%% imagecomplex
function imagecomplex(varargin)
% Get screensize and get figure
scrsz = get(0,'ScreenSize');
figure('Position',[(scrsz(3)-1280)/2+1 (scrsz(4)-720)/2 1280 720],'color','w');

subplot(2,1,1)
imagesc(abs(varargin{1}))
colormap(jet)
title('abs(image)')
colorbar
axis image xy

subplot(2,1,2)
imagesc(angle(varargin{1}))
colormap(jet)
title('angle(image)')
colorbar
axis image xy
