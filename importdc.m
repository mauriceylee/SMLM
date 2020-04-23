function dark_counts = importdc(varargin)
% Import DARK COUNTS and average them as a single frame
[dataFile1, dataPath] = uigetfile({'*.tif';'*.*'},'Open file for DARK COUNTS');

if isequal(dataFile1,0), error('User cancelled the program'); end

% Get some info about the dark counts flie
dataFile = [dataPath dataFile1];
dataFileInfo = imfinfo(dataFile);
numFrames = length(dataFileInfo);
imgHeight = dataFileInfo.Height;    
imgWidth = dataFileInfo.Width;

%% Sum all the frames of the dark counts first
im = tiffread2(dataFile,1,numFrames);
allthedata = zeros(imgHeight,imgWidth,numFrames);
for i = 1:numFrames
    allthedata(:,:,i) = double(im(i).data);
end
% darkcounts will be a 2D matrix that consists of the mean of each pixel
dark_counts = mean(allthedata,3);

%% Check out the dark counts
figmau
subplot(1,2,1)
imagesc(dark_counts)
axis xy image
% colormap(linearcolormap)
colorbar
title(['Dark counts. Average of ',num2str(numFrames),' frames.'])

xlabel('X (pixels)')
ylabel('Y (pixels)')
subplot(1,2,2)
hist(dark_counts(:),100)
xlabel('Average intensity of dark counts (raw pixel counts)')
fig2pretty

subplot(1,2,1)
grid off
end