% sPSF_SM_blinking_v34_forGithub.m
% Written by Maurice Lee

% This script analyzes SMLM data and is very messy!
% It converts the tif files with single-molecule blinking data into a table
% of important parameters.

% There is background estimation, astigmatic Gaussian fitting, filtering 
% of poor fits, combining of ROIs before a second round of Gaussian fit.

%% Welcome message and report the starting date and time
fprintf('Hello! Welcome to analyzing your single-molecule blinking data!\n')
ynt = clock;
fprintf(['Starting date and time: ', ...
    num2str(ynt(1)), num2str(ynt(2), '%02d'), num2str(ynt(3), '%02d'), ' ', ...
    num2str(ynt(4)), ':', num2str(ynt(5), '%02d'), ' hrs\n'])

load inferno.mat

%% Key in some input parameters that are important
experiment_Prompt = {'Pixel size [nm]', ...
    'ADC', ...
    'EM Gain', ...
    'Theta of astigmatic fit [degrees] (Use 0 to test all angles. If not, just use 45)',...
    'Peak ellipticity threshold [0.15]',...
    'ROIsize (leave at 7 for pixel size = 160 nm)', ...
    'Initial guess for lower limit for sigma [0.68 for 647, 0.7 for 561]', ...
    'Initial guess for higher limit for sigma [0.84->0.90 for 647, 0.82 for 561]', ...
    'Number of off frames [Try 5 to 10 first]', ...
    'Min. expected #photons per molecule [1000]. Just use 0 to calculate it',...
    'Background estimation: Temporal [0], Spatial [1]'};
    
defaultExperimentInputs = {'160', ...
    '27.4', ...
    '184', ...
    '45',...
    '0.15',...
    '7', ...
    '0.68', ...
    '0.90', ...
    '5', ...
    '0', ...
    '1'};

experimentInputs = inputdlg(experiment_Prompt,'Input experimental parameters',1,defaultExperimentInputs);

pixelsizenm = str2double(experimentInputs{1}); % pixel size may change for different color channels
ADC = str2double(experimentInputs{2}); % this analog to digital conversion was calibrated via the Thompson paper
gain = str2double(experimentInputs{3}); % gain is the setting you chose while imaging the cells

Thetafixed = str2double(experimentInputs{4}); % 45 degrees was chosen for our microscope after we allow it to vary from 0 to 90 degrees
% Decide if theta is fixed or not
if Thetafixed == 0
    thetalimlow = 0;
    thetalimhigh = 90;
    thetastart = 45;
else
    thetalimlow = Thetafixed-0.001;
    thetalimhigh = Thetafixed+0.001;
    thetastart = Thetafixed;
end

ellipticity_threshold = str2double(experimentInputs{5});
ROIsize = str2double(experimentInputs{6});
ROIhalfwidth = floor(ROIsize/2);

sigmax_threshold = [str2double(experimentInputs{7}) str2double(experimentInputs{8})];
sigmay_threshold = [str2double(experimentInputs{7}) str2double(experimentInputs{8})];

totalnumberoffframes = str2double(experimentInputs{9});

actual_threshold_for_filtered_images = str2double(experimentInputs{10});

usespatialfilter = str2double(experimentInputs{11}); % 1 for using spatial filter, 0 for using temporal filter

fprintf(['Pixel size = ',num2str(pixelsizenm),' nm\n'])
fprintf(['ADC = ',num2str(ADC),'\n'])
fprintf(['EM gain = ',num2str(gain),'\n'])
if Thetafixed == 0
    fprintf('Theta is allowed to vary from 0 to 90 degrees\n')
else
    fprintf(['Theta is fixed at ',num2str(Thetafixed),' degrees\n'])
end
fprintf(['Peak Ellipticity threshold = ',num2str(ellipticity_threshold),'\n'])
fprintf(['ROI size is ',num2str(ROIsize),'x',num2str(ROIsize),' pixels\n'])
fprintf(['Thresholds for sigma are [',num2str(sigmax_threshold(1)),' ',num2str(sigmax_threshold(2)),']\n\n'])
fprintf(['Number of allowed off frames = ',num2str(totalnumberoffframes),'\n\n'])
fprintf(['Estimated MIN. #photons per molecule = ',num2str(actual_threshold_for_filtered_images),'\n\n'])

% options for nonlinear least square fitting of Gaussian later
options = optimset('TolFun',1e-6, 'MaxFunEvals', 3000, 'MaxIter', 3000,'Display','off','FunValCheck','on');
    

%% Calculate average dark counts
% Import dark counts that have been cropped in imageJ
dark_counts = importdc; % importdc does the averaging into a 2D frame
%  dark_counts = zeros(64,64); 





%% Import files for main blinking data
[dataFile1, dataPath] = uigetfile({'*.tif';'*.*'},'Open file for main file');
if isequal(dataFile1,0), error('User cancelled the program'); end
% Get some info about the dark counts flie
dataFile = [dataPath dataFile1];
disp(['File name = ', (dataFile)])
dataFileInfo = imfinfo(dataFile,'tif');
total_num_frames = length(dataFileInfo);
imgHeight = dataFileInfo.Height;
imgWidth = dataFileInfo.Width;

%% Create a new analysis folder in the folder where the data was taken from
YMD = [num2str(ynt(1)) num2str(ynt(2),'%02d') num2str(ynt(3),'%02d') num2str(ynt(4),'%02d') num2str(ynt(5),'%02d')];
analysisFolderName = [YMD '_' dataFile1(1:end-4) '_AnalysisFolder'];
pathtoanalysisfolder = [dataPath analysisFolderName];
[status,~] = mkdir(pathtoanalysisfolder);

%% Create a textfile to save fprintf
S = dbstack();
thisscriptname = S(1).file;
thisscriptname = thisscriptname(1:end-2);
thisscriptname = char(thisscriptname);
commandwindowlogfilename = [dataFile1(1:end-4) '_' thisscriptname 'analysis.txt'];

%% Start txt file for analysis
fileID = fopen(commandwindowlogfilename,'w+');

%% Write some things to the text file
fprintf(fileID,'Hello! Welcome to analyzing your single-molecule blinking data!\r\n');
fprintf(fileID,['Starting date and time: ', ...
    num2str(ynt(1)),num2str(ynt(2),'%02d'),num2str(ynt(3),'%02d'),' ', ...
    num2str(ynt(4)),':',num2str(ynt(5),'%02d'),' hrs\r\n']);
fprintf(fileID,['Pixel size = ',num2str(pixelsizenm),' nm\r\n']);
fprintf(fileID,['ADC = ',num2str(ADC),'\r\n']);
fprintf(fileID,['EM gain = ',num2str(gain),'\r\n']);
if Thetafixed == 0
    fprintf(fileID,'Theta is allowed to vary from 0 to 90 degrees\r\n');
else
    fprintf(fileID,['Theta is fixed at ',num2str(Thetafixed),' degrees\r\n']);
end
fprintf(fileID,['Peak Ellipticity threshold = ',num2str(ellipticity_threshold),'\r\n']);
fprintf(fileID,['ROI size is ',num2str(ROIsize),'x',num2str(ROIsize),' pixels\r\n']);
fprintf(fileID,['Thresholds for sigma are [',num2str(sigmax_threshold(1)),' ',num2str(sigmax_threshold(2)),']\r\n']);
if usespatialfilter == 1
    fprintf(fileID,'We are using a spatial filter for background estimation\r\n');
else
    fprintf(fileID,'We are using a temporal filter for background estimation\r\n');
end
fprintf(fileID,['Number of allowed off frames = ',num2str(totalnumberoffframes),'\r\n\r\n']);


%% Import all the data
im = tiffread22(dataFile,1,total_num_frames); % tiffread22 is actually tiffread with some things commented out
DATA = zeros(imgHeight,imgWidth,total_num_frames);
for i = 1:total_num_frames
        DATA(:,:,i) = double(im(i).data);
end

% These frames are usually cropped and concatenated in imageJ before import
DATA = single(DATA); % single uses less memory than double

fprintf('DATA has been imported in workspace.\n\n')
fprintf(['Size of DATA is ',num2str(size(DATA,1)),' x ',num2str(size(DATA,2)),' x ',num2str(total_num_frames),'.\n\n'])
fprintf(fileID,'DATA has been imported in workspace.\r\n');
fprintf(fileID,['Size of DATA is ',num2str(size(DATA,1)),' x ',num2str(size(DATA,2)),' x ',num2str(total_num_frames),'.\r\n']);

%% Remove a fixed offset if need be, skip this if you don't want to use this
if exist('dark_counts','var') == 1
    % Subtract darkcounts
    for i = 1:size(DATA,3)
        DATA(:,:,i) = DATA(:,:,i) - dark_counts;
    end
    fprintf('Dark counts have been subtracted.\n')
    fprintf(fileID,'Dark counts have been subtracted.\r\n');
end

% % % DATA = DATA - min(DATA(:));

%%
clear dark_counts im dataFileInfo

%% Convert to photon counts with gain and ADC
DATA = DATA./gain.*ADC;

fprintf(['DATA has been converted from AU to photons with ADC = ',num2str(ADC),' and gain = ',num2str(gain),'.\n\n'])
fprintf(fileID,['DATA has been converted from AU to photons with ADC = ',num2str(ADC),' and gain = ',num2str(gain),'.\r\n\r\n']);

%% Check first frame
firstframe = DATA(:,:,1);
imagescmau(firstframe)
title('First frame')
xlabel('X (pixels)')
ylabel('Y (pixels)')
c = colorbar;
c.Label.String = 'Photons';
fig2pretty
grid off
axis image xy

%% Save figure
figurecounter = 1;
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% Estimate background
 
if usespatialfilter == 1
    %% Use temporal LSP filter. LSP = local statistical percentile
    % Use the temporal filter only if the blinking density is low enough
    % (i.e. the out-of-focus background from blinking molecules doesn't change much from frame-to-frame)
    
    fprintf('Starting to estimate temporal median background.\n');
    fprintf(['This may take ',num2str(numel(DATA)/4.1937E6/60,3),' mins or more.\n']);
    fprintf(fileID,'Starting to estimate temporal median background.\r\n');
    fprintf(fileID,['This may take ',num2str(numel(DATA)/4.1937E6/60,3),' mins or more.\r\n']);
    
    %% Flip DATA such that time is in the first dimension (i.e. flip it from 1 2 3 to 3 2 1)
    DATA_flip = permute(DATA,[3 2 1]);
    background_matrix = zeros(size(DATA_flip,1),size(DATA_flip,2),size(DATA_flip,3)); % make a matrix that is the same size as allthedata_test
    
    %% Parameters for temporal median filtering
    window_size = 51; % this is the size of the window to calculate the temporal LSF filter
    fprintf(['Temporal median window size = ',num2str(window_size),'.\n']);
    bintouse = ones(window_size,1);
    Nthdimmestpixel = floor(window_size/2);
    % N-th dimmest pixel in the temporal bin, arbitrary number, half or smaller would sort of work
    
    %%
    tic
    DATA_flip = double(DATA_flip);
    parfor whichframe = 1:size(DATA_flip,3)
        I = DATA_flip(:,:,whichframe);
        I_filtered = ordfilt2(I,Nthdimmestpixel,bintouse,'symmetric'); % median filter in temporal domain
        background_matrix(:,:,whichframe) = I_filtered;
    end
    background_matrix = single(background_matrix);
    background_matrix = permute(background_matrix,[3 2 1]);
    toc
    
    %%
    clear DATA_flip 
    
    %%
%     whichframe = 205;
%     DATAframe = DATA(:,:,whichframe);
%     bgframe = background_matrix(:,:,whichframe);
%     DATAwobgframe = DATA(:,:,whichframe) - bgframe;
%     imagescmau(DATAframe, DATAwobgframe, bgframe)
%    
%     figmau
%     ax1 = subplot(1,3,1);
%     imagesc(DATAframe)
%     title('Original data')
%     axis image xy
%     colorbar
%     colormap(inferno)
%     
%     ax2 = subplot(1,3,2);
%     imagesc(bgframe)
%     title('Estimated bg')
%     xlabel(['Frame ',num2str(whichframe)])
%     axis image xy
%     colormap(inferno)
%     colorbar
%     caxis([min(min(DATA(:,:,whichframe))) max(max(DATA(:,:,whichframe)))])
%     
%     ax3 = subplot(1,3,3);
%     imagesc(DATAwobgframe)
%     title('After removing bg')
%     axis image xy
%     c = colorbar;
%     colormap(inferno)
%     c.Label.String = 'Photons';
%     
%     linkaxes([ax1,ax2,ax3],'xy')

else
    
    %% SPATIAL FILTER for background estimation
    fprintf('Starting to estimate spatial background for each frame...\n');
    fprintf(['This may take ',num2str(size(DATA,3)/198/60,3),' mins or more.\n']);
    fprintf(fileID,'Starting to estimate spatial background for each frame...\r\n');
    fprintf(fileID,['This may take ',num2str(size(DATA,3)/40/60,3),' mins or more.\r\n']);
    
    background_matrix = zeros(size(DATA,1),size(DATA,2),size(DATA,3)); % make a matrix that is the same size as allthedata_test

    %% Choose parameters for spatial filtering
    bintouse = ones(ROIsize,ROIsize);
    Nthdimmestpixel = 14;
    
    %%
    tic
    DATA = double(DATA);
    for whichframe = 1:size(DATA,3)
        I = DATA(:,:,whichframe);
        I_filtered = ordfilt2(I,Nthdimmestpixel,bintouse,'symmetric'); % the 14th dimmest pixel in a 7x7 ROI is the background
%         I_filtered = imgaussfilt(I_filtered,2,'FilterDomain','spatial'); % sigma of 2 is arbitrary for the Gaussian smoothing of the background estimate
        % note. lambda/2/NA = 700/2/1.4 = 250 nm = 1.56 pixels. so sigma =
        % 2 is OK. But it's unsure if a Gaussian smoothing is needed
        background_matrix(:,:,whichframe) = I_filtered;
    end
    DATA = single(DATA);
    background_matrix = single(background_matrix);
    toc
    
    %%
% % % %     DATAwobgframe = I-I_filtered;
% % % %     imagescmau(I, DATAwobgframe, I_filtered)
% % % %     %     caxis([0 100])
% % % %     
% % % %     
% % % %     whichframe = 3000;
% % % %     figmau
% % % %     ax1 = subplot(1,3,1);
% % % %     imagesc(DATA(:,:,whichframe))
% % % %     title('Original data')
% % % %     axis image xy
% % % %     colorbar
% % % %     colormap(inferno)
% % % %     
% % % %     ax2 = subplot(1,3,2);
% % % %     imagesc(I_filtered)
% % % %     title('Estimated bg')
% % % %     xlabel(['Frame ',num2str(whichframe)])
% % % %     axis image xy
% % % %     colormap(inferno)
% % % %     colorbar
% % % %     caxis([min(min(DATA(:,:,whichframe))) max(max(DATA(:,:,whichframe)))])
% % % %     
% % % %     ax3 = subplot(1,3,3);
% % % %     imagesc(DATAwobgframe)
% % % %     title('After removing bg')
% % % %     axis image xy
% % % %     c = colorbar;
% % % %     colormap(inferno)
% % % %     c.Label.String = 'Photons';
% % % %     
% % % %     linkaxes([ax1,ax2,ax3],'xy')

    %%
    clear I I_filtered
    
end
fprintf('Background has been estimated.\n\n');
fprintf(fileID,'Background has been estimated.\r\n');

%% Remove background
DATA_wobg = DATA - background_matrix;
fprintf('Estimated background has been removed.\n\n');
fprintf(fileID,'Estimated background has been removed.\r\n\r\n');

%% Plot background intensity in whole ROI with time
figmau
plot(squeeze(sum(sum(background_matrix))))
ylabel('Background photons in entire FOV')
xlabel('Frame number')
fig2pretty

%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% save the stack as a tif file for thunderstorm analysis
% THIS TAKES HOURS, skip this if need be
% for i = 1:size(data_wobg,3)
%     imwrite(uint16(data_wobg(:,:,i)),'sequence-as-stack-MT0.N1.LD-2D-Exp.tif','writemode','append');
% end

%% Look at frames before and after removing the background

whichframe = 125;
figmau
ax1 = subplot(1,3,1);
imagesc(DATA(:,:,whichframe))
title('Original data')
axis image xy
colorbar
colormap(inferno)

ax2 = subplot(1,3,2);
imagesc(background_matrix(:,:,whichframe))
title('Estimated bg')
xlabel(['Frame ',num2str(whichframe)])
axis image xy
colormap(inferno)
colorbar
% caxis([min(min(DATA(:,:,whichframe))) max(max(DATA(:,:,whichframe)))])

ax3 = subplot(1,3,3);
imagesc(DATA_wobg(:,:,whichframe))
title('After removing bg')
axis image xy
c = colorbar;
colormap(inferno)
c.Label.String = 'Photons';

linkaxes([ax1,ax2,ax3],'xy')


%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%%
figmau
subplot(1,2,1)
imagesc(sum(background_matrix(:,:,1000:2000),3))
grid off
title('Sum of background')
xlabel('Frames 2000 to 3000')
fig2pretty

subplot(1,2,2)
imagesc(sum(background_matrix(:,:,4100:4200),3))
grid off
title('Sum of background')
xlabel('Frames 41000 to 42000')
fig2pretty

%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% Look at one spot in one frame
DATAwobgframe = DATA_wobg(:,:,whichframe);
[y589,x589] = find(DATAwobgframe==max(DATAwobgframe(:)));

x589 = round(x589); y589 = round(y589);

ROI1 = DATA(y589-ROIhalfwidth:y589+ROIhalfwidth,x589-ROIhalfwidth:x589+ROIhalfwidth,whichframe);
ROI2 = DATA_wobg(y589-ROIhalfwidth:y589+ROIhalfwidth,x589-ROIhalfwidth:x589+ROIhalfwidth,whichframe);
ROI3 = background_matrix(y589-ROIhalfwidth:y589+ROIhalfwidth,x589-ROIhalfwidth:x589+ROIhalfwidth,whichframe);

% ROI1 = testframe(y589-3:y589+3,x589-3:x589+3);
% ROI2 = DATAwobgframe(y589-3:y589+3,x589-3:x589+3);
% ROI3 = testframe_filtered(y589-3:y589+3,x589-3:x589+3);

figmau
subplot(1,3,1)
imagesc(ROI1)
axis image xy
colorbar
title('Before')
xlabel('Brightest localization')
axis image xy

subplot(1,3,2)
imagesc(ROI3)
axis image xy
c = colorbar;
c.Label.String = 'Photons';
caxis([min(min(ROI1)) max(max(ROI1))])
title('Bg')

subplot(1,3,3)
imagesc(ROI2)
axis image xy
colorbar
title('After')
xlabel(['Frame #',num2str(whichframe)])
axis image xy
fig2pretty
colormap(inferno)

%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%%
clear background_matrix




%%

%% Choose a threshold for diff abs
if actual_threshold_for_filtered_images == 0
    %% CHOOSE THRESHOLD to identify spots
    SCALINGFORTHRESHOLD = 1; %% <--- change this if need be
    stdINT = median(DATA_wobg(:))-min(DATA_wobg(:));
    medianINT = median(DATA_wobg(:));
    diffabsthreshold = medianINT + stdINT * SCALINGFORTHRESHOLD;
    
    %% Look at images after thresholding
    imagescmau(DATA_wobg(:,:,100), DATA_wobg(:,:,100)>diffabsthreshold)
    title(['Frame 100. Threshold is ',num2str(round(diffabsthreshold))])
    
    figurecounter = savefigure(figurecounter,pathtoanalysisfolder);
    
else
    diffabsthreshold = actual_threshold_for_filtered_images;
end

clear DATA_diff_abs

fprintf(['The threshold we are going to use is ',num2str(round(diffabsthreshold)),' photons.\n']);
fprintf(['Placing the threshold on each of the ',num2str(size(DATA_wobg,3)),' frames now...\n']);

fprintf(fileID,['The threshold we are going to use is ',num2str(round(diffabsthreshold)),' photons.\r\n']);
fprintf(fileID,['Placing the threshold on each of the ',num2str(size(DATA_wobg,3)),' frames now...\r\n']);


%% Create binary masks for brightspots
DATA_wobg_BW = DATA_wobg>diffabsthreshold;

%%
imagescmau(DATA_wobg_BW(:,:,whichframe) , DATA(:,:,whichframe))
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% Find each spot , make ROI , check ROI before or after
expectednumber = size(DATA_wobg_BW,3)*50; % estimate 50 localizations per frame
WC_x = zeros(expectednumber,1,'single');
WC_y = zeros(expectednumber,1,'single');
FrameNumber = zeros(expectednumber,1,'single');
counter = 1;

% Need to calculate Frame number and Weighted centroid for each localization
for thisframe = 1:size(DATA_wobg_BW,3) % 15828 %
    CC = bwconncomp(DATA_wobg_BW(:,:,thisframe),8); % find which pixels are connected, 8 connectivity
    stats = regionprops(CC,DATA(:,:,thisframe),'WeightedCentroid','Area');
    AreaOfBlobs = cat(1,stats.Area);
    stats(AreaOfBlobs<4) = []; % remove spots fewer than 4 pixels, 4 pixels is arbitrary
    
    WCOfBlobs = cat(1,stats.WeightedCentroid); % find weighted centroids
    if ~isempty (WCOfBlobs)
        WCOfBlobs_x = WCOfBlobs(:,1);
        WCOfBlobs_y = WCOfBlobs(:,2);
        
        NumberOfBlobs = size(stats,1);
        
        % save weighted centroids and frame numbers
        WC_x(counter:counter+NumberOfBlobs-1) = WCOfBlobs_x;
        WC_y(counter:counter+NumberOfBlobs-1) = WCOfBlobs_y;
        FrameNumber(counter:counter+NumberOfBlobs-1) = thisframe;
        
        counter = counter+NumberOfBlobs;
    end
end

%% Remove additional zeroes at end of vector
WC_x = WC_x(1:counter-1);
WC_y = WC_y(1:counter-1);
FrameNumber = FrameNumber(1:counter-1);

% Round WC_x and WC_y to draw ROI later
WC_x_round = round(WC_x);
WC_y_round = round(WC_y);

fprintf('Done with placing threshold!\n\n');
fprintf(fileID,'Done with placing threshold!\r\n\r\n');

%% clear some data
clear DATA_PSF_BW

% sum the frames
sumofallframes = sum(DATA_wobg,3);

%% Check out all the localizations thus far
figmau
plot(WC_x,WC_y,'k.','MarkerSize',4)
axis image xy
title('Scatter plot of weighted centroids in all frames')
fig2pretty

h = hist3([WC_y WC_x],[size(DATA_wobg,1)*5 size(DATA_wobg,2)*5]);
imagescmau(h)
clear h
axis image xy
title('Histogram of all weighted centroids')
fig2pretty
% caxis([0 10])

imagescmau(sumofallframes)
axis image xy
title('Histogram of sum of all frames without background')
fig2pretty
% caxis([0 2.5E6])

%% FIRST ROUND OF FITTING WITH 2D ASYMMETRIC GAUSSIAN
num_of_localizations = length(WC_x);
fprintf(['There are currently ',num2str(num_of_localizations),' localizations detected at the very start.\n\n']);
fprintf(fileID,['There are currently ',num2str(num_of_localizations),' localizations detected at the very start.\r\n']);

% Fit asymetric Gaussian with offset as a variable
output_matrix = zeros(num_of_localizations,14,'single');

% 1 == molecule_ID
% 2 == X (nm)
% 3 == Y (nm)
% 4 == sigmaX
% 5 == sigmaY
% 6 == theta
% 7 == totalphotons
% 8 == offset
% 9 == meanbackground (from earlier estimation)
% 10 == mean residual
% 11 == max residual
% 12 == frame first appeared
% 13 == total number of frames
% 14 == OK_result

fprintf(['We are going to crop each localization into an ROI with ',num2str(ROIsize), ...
    'x',num2str(ROIsize),' pixels.\n\n'])
fprintf(fileID,['We are going to crop each localization into an ROI with ',num2str(ROIsize), ...
    'x',num2str(ROIsize),' pixels.\r\n']);

%% Initiate some variables for each molecule first
InitialTotalIntensityGuess_Log = zeros(num_of_localizations,1,'single');
InitialDimmestPixelGuess_Log = zeros(num_of_localizations,1,'single');
ROItofit = zeros(ROIsize,ROIsize,num_of_localizations,'single');
OK_result = true(num_of_localizations,1);
width_DATA = size(DATA_wobg,2);
height_DATA = size(DATA_wobg,1);

% Crop each localization into an ROI
for loc_ID = 1:num_of_localizations
    % Round the weighted centroid of x and y
    mid_x = WC_x(loc_ID);
    mid_y = WC_y(loc_ID);
    mid_x_round = WC_x_round(loc_ID);
    mid_y_round = WC_y_round(loc_ID);
    
    %% Check if 7x7 ROI is within the entire cropped area
    if mid_x_round > ROIhalfwidth+1 && mid_x_round <= width_DATA-ROIhalfwidth-1 && ...
            mid_y_round > ROIhalfwidth+1 && mid_y_round <= height_DATA-ROIhalfwidth-1
        %% pixels in cropped area to become ROI
        pixels_x = mid_x_round-ROIhalfwidth:mid_x_round+ROIhalfwidth;
        pixels_y = mid_y_round-ROIhalfwidth:mid_y_round+ROIhalfwidth;
        ROItofit(:,:,loc_ID) = DATA_wobg(pixels_y,pixels_x, FrameNumber(loc_ID));
        InitialTotalIntensityGuess_Log(loc_ID) = sum(sum(ROItofit(:,:,loc_ID))); 
        % 0.6 is an empirical number of totalphotons over sum of all photons
        InitialDimmestPixelGuess_Log(loc_ID) = min(min(ROItofit(:,:,loc_ID)));
        
        %% Calculate background first
        dataframe = DATA(pixels_y,pixels_x, FrameNumber(loc_ID));
        backgroundframe = dataframe - ROItofit(:,:,loc_ID);
        B = sum(sum( backgroundframe ))/(ROIsize^2); % from median background
        output_matrix(loc_ID,9) = B; % photons per pixel
        
        %% frame number for this molecule
        output_matrix(loc_ID,12) = FrameNumber(loc_ID);
        
    else % if ROI is too near edge of cropped area
        OK_result(loc_ID) = 0;
    end
end
background = output_matrix(:,9);

fprintf('Each ROI is now cropped.\n\n')
fprintf(fileID,'Each ROI is now cropped.\r\n');

% Initialize some vectors to feed into the parfor loop

X_pixels = zeros(num_of_localizations,1,'single'); % column 2
Y_pixels = zeros(num_of_localizations,1,'single'); % column 3
sigmaX = zeros(num_of_localizations,1,'single'); % column 4
% sigmaY = zeros(num_of_localizations,1,'single'); % column 5
thetavec = zeros(num_of_localizations,1,'single'); % column 6
totalphotons = zeros(num_of_localizations,1,'single'); % column 7
offsetvec = zeros(num_of_localizations,1,'single'); % column 8
meanresidual = zeros(num_of_localizations,1,'single'); % column 10
maxresidual = zeros(num_of_localizations,1,'single'); % column 11

%% Fit the 2D asymmetric Gaussian for each slice in ROItofit
fprintf('We are going to fit each localization with a 2D Gaussian.\n')
fprintf(['This will take approx. ',num2str(ceil(num_of_localizations/238/60)),' mins with 4 cores.\n\n'])
fprintf(fileID,'We are going to fit each localization with a 2D Gaussian.\r\n');
fprintf(fileID,['This will take approx. ',num2str(ceil(num_of_localizations/238/60)),' mins with 4 cores.\r\n\r\n']);

clear par fitparam residual
[ii,jj] = meshgrid(1:ROIsize,1:ROIsize);
sigmadelta = 0.05;
sigmaXlowerthreshold = sigmax_threshold(1)-sigmadelta;
% sigmaYlowerthreshold = sigmay_threshold(1)-sigmadelta;
sigmaXhigherthreshold = sigmax_threshold(2)+sigmadelta;
% sigmaYhigherthreshold = sigmay_threshold(2)+sigmadelta;
maxXpixelinROI = ROIsize-1;
maxYpixelinROI = ROIsize-1;
Xpixelinitialguess = (WC_x-WC_x_round)+ceil(ROIsize/2);
Ypixelinitialguess = (WC_y-WC_y_round)+ceil(ROIsize/2);
meansigmaXguess = mean(sigmax_threshold);
meansigmaYguess = mean(sigmay_threshold);

tic
parfor loc_ID = 1:num_of_localizations
    if OK_result(loc_ID)==1 % fit only if the localization is far from edge of cropped area

        %% Fit the data with a 2D asymmetric Gaussian
%             loc_ID = 5000;
        ROItofitnow = ROItofit(:,:,loc_ID);
        InitialIntensityGuess = InitialTotalIntensityGuess_Log(loc_ID); % initial guess for amplitude
        min_ROItofitnow = InitialDimmestPixelGuess_Log(loc_ID); % initial guess for dimmest pixel
        
         % asymmetric Gaussian fit
    %     lowerBound = [500; 2; 2; sigmaXlowerthreshold; sigmaYlowerthreshold; thetalimlow; min_ROItofitnow];
    %     upperBound = [InitialIntensityGuess; maxXpixelinROI; maxYpixelinROI; ...
    %         sigmaXhigherthreshold; sigmaYhigherthreshold; thetalimhigh; InitialIntensityGuess];
    %     DC = DC_log(loc_ID) + 30;
    
    % % % %     [fitparam,~,residual,~,~,~,~] = ...
    % % % %         lsqnonlin(@(par) singleIntegratedGaussianOffset( par,double(newROItofitnow),ii,jj), ...
    % % % %         double([0.6*InitialIntensityGuess; XYpixelinitialguess; XYpixelinitialguess; meansigmaXguess; meansigmaYguess; thetastart; DC]), ...
    % % % %         double(lowerBound), ...
    % % % %         double(upperBound),options);
    
    % symmetric Gaussian fit
    lowerBound = [500; 2; 2; sigmaXlowerthreshold; meansigmaYguess; thetastart; min_ROItofitnow];
    upperBound = [InitialIntensityGuess; maxXpixelinROI; maxYpixelinROI; ...
        sigmaXhigherthreshold; meansigmaYguess; thetastart; InitialIntensityGuess];
    DC = min_ROItofitnow + 30;
    
    [fitparam,~,residual,~,~,~,~] = ...
        lsqnonlin(@(par) singleIntegratedGaussianOffset_symmetricversion( par,double(ROItofitnow),ii,jj), ...
        double([0.6*InitialIntensityGuess; Xpixelinitialguess(loc_ID); Ypixelinitialguess(loc_ID); meansigmaXguess; meansigmaYguess; thetastart; DC]), ...
        double(lowerBound), ...
        double(upperBound),options);
    
    %   par(1) = intensity
    %   par(2) = Xo
    %   par(3) = Yo
    %   par(4) = sigmaX;
    %   par(5) = sigmaY;
    %   par(6) = theta; degrees
    %   par(7) = DC;
    
    sum_sum_abs_residual = sum(abs(residual(:)));
        
        %% Troubleshoot Gaussian fit
        

% % %         figmau
% % %         subplot(2,2,1)
% % %         imagesc(ROItofitnow)
% % %         title(num2str(loc_ID))
% % %         colorbar
% % %         axis image xy
% % %         
% % %         subplot(2,2,2)
% % %         par = fitparam;
% % % % % % % %         theta = par(6)*pi/180;
% % % % % % % %         E_x = 0.5*( erf( ((ii-par(2))*cos(theta) + (jj-par(3))*sin(theta) + 0.5) / (2*par(4)*par(4))) - ...
% % % % % % % %             erf( ((ii-par(2))*cos(theta) + (jj-par(3))*sin(theta) - 0.5) / (2*par(4)*par(4))));
% % % % % % % %         E_y = 0.5 * ( erf( (-(ii-par(2))*sin(theta) + (jj-par(3))*cos(theta) + 0.5) / (2*par(5)*par(5))) - ...
% % % % % % % %             erf( (-(ii-par(2))*sin(theta) + (jj-par(3))*cos(theta) - 0.5) / (2*par(5)*par(5))));
% % % % % % % %         mu = par(1).*E_x.*E_y + par(7);
% % %         
% % %         % symmetric
% % %         E_x = 0.5 * ( erf( (ii-par(2)+0.5) / (2*par(4)*par(4)) ) - erf( (ii-par(2)-0.5) / (2*par(4)*par(4)) ));
% % %         E_y = 0.5 * ( erf( (jj-par(3)+0.5) / (2*par(4)*par(4)) ) - erf( (jj-par(3)-0.5) / (2*par(4)*par(4)) ));
% % %         mu = par(1).*E_x.*E_y + par(7);
% % %         imagesc(mu)
% % %         axis image xy
% % %         colorbar
% % %         
% % %         subplot(2,2,4)
% % %         imagesc(reshape(residual,size(mu)))
% % %         xlabel(['Sum of residual=',num2str(sum_sum_abs_residual )])
% % %         title(['theta = ',num2str(fitparam(6)/pi*180,3),' degrees'])
% % %         colorbar
% % %         axis image xy

        %% Outputs
        totalphotons(loc_ID) = fitparam(1); % total photons
        X_pixels(loc_ID) = fitparam(2)+WC_x_round(loc_ID)-ROIhalfwidth-1; % x location, pixels
        Y_pixels(loc_ID) = fitparam(3)+WC_y_round(loc_ID)-ROIhalfwidth-1; % y location, pixels
        sigmaX(loc_ID) = fitparam(4); % sigma X, pixels
%         sigmaY(loc_ID) = fitparam(5); % sigma Y, pixels
%         thetavec(loc_ID) = fitparam(6); % theta, degrees
        meanresidual(loc_ID) = sum_sum_abs_residual/ROIsize/ROIsize; % residual per pixel
        if ~isempty(residual)
            maxresidual(loc_ID) = max(abs(residual(:))); % max residual
        end
        offsetvec(loc_ID) = fitparam(7); % flat offset, photons per pixel
        
    end
end
toc

%% Place the vectors from the parallel loop into a big table

X_nm = (X_pixels-0.5).*pixelsizenm; % X in nm
Y_nm = (Y_pixels-0.5).*pixelsizenm; % Y in nm
output_matrix(:,2) = X_nm; % x location, nm
output_matrix(:,3) = Y_nm; % y location, nm
output_matrix(:,4) = sigmaX; % sigma X, pixel
% output_matrix(:,5) = sigmaY; % sigma Y, pixel
output_matrix(:,5) = sigmaX; % sigma Y, pixel
sigmaY = sigmaX;
output_matrix(:,6) = thetavec; % theta, degrees
output_matrix(:,7) = totalphotons; % total photons
output_matrix(:,8) = offsetvec; % photons per pixel
output_matrix(:,10) = meanresidual; % residual
output_matrix(:,11) = maxresidual;

% 1 == molecule_ID
% 2 == X (nm)
% 3 == Y (nm)
% 4 == sigmaX
% 5 == sigmaY
% 6 == theta
% 7 == totalphotons
% 8 == offset
% 9 == meanbackground (from earlier estimation)
% 10 == mean residual
% 11 == max residual
% 12 == frame first appeared
% 13 == total number of frames, this is left blank in the first
% output_matrix because we only know the frame length after combining
% localizations into molecules
% 14 == final_OK

%% Scatter plot of centroids
figmau
plot(output_matrix(:,2), ...
    output_matrix(:,3) ,'k.');
axis image xy
xlabel('X (nm)')
ylabel('Y (nm)')
title('Scatter of fitted centroids')
fig2pretty


%% Need to filter results

%% How filtering results work
% 1a. Remove multi-emitters with sigmaX and sigmaY and peak ellipiticity
% 1b. Look at histogram of residual to remove multi-emitters (exclude those
% that were removed by step 1a.)
% 2. Remove dim, imprecise localizations with photon count, background
% +offset and Mortensen precision

fprintf(['\nAfter first round of 2D Gaussian fits, we start with #OK localizations = ',num2str(sum(OK_result)),'\n\n'])
fprintf(fileID,['\nAfter first round of 2D Gaussian fits, we start with #OK localizations = ',num2str(sum(OK_result)),'\r\n\r\n']);

%% HISTOGRAM three thresholds on each of the 1D histogram for Sigma_X, Sigma_Y and Peak_Ellipticity 

Peak_Ellipticity = 2*(sigmaX-sigmaY)./(sigmaX+sigmaY); % Peak ellipticity formula (XWZ nat meth 2006)

% Peak_Ellipticity
figmau
hist(Peak_Ellipticity,200)
xlabel('Peak Ellipticity = 2*(A-B)/(A+B)')
title(['Threshold = ',num2str(ellipticity_threshold)])
hold on
plot([ellipticity_threshold ellipticity_threshold], [0 500],'r')
plot([-ellipticity_threshold -ellipticity_threshold ], [0 500],'r')
hold off
fig2pretty

%% Find lower and upper thresholds for sigma_X
numberofsigma = 2.2; % <<<---- this is empirical and what i found to work well
[sigmaX_calc_highthreshold,sigmaX_calc_lowthreshold,sigmaX_logicalwithinthresh] = sigmaclip2tail(sigmaX(sigmaX>0),numberofsigma);

figmau
hist(sigmaX,200)
xlabel('Sigma X (pixels)')
hold on
plot([sigmaX_calc_lowthreshold sigmaX_calc_lowthreshold], [0 500],'r')
plot([sigmaX_calc_highthreshold sigmaX_calc_highthreshold], [0 500],'r')
hold off
title(['Threshold = [',num2str(sigmaX_calc_lowthreshold), ...
    ' ',num2str(sigmaX_calc_highthreshold),']'])
fig2pretty

%% Find lower and upper thresholds for sigma_Y
[sigmaY_calc_highthreshold,sigmaY_calc_lowthreshold,sigmaY_logicalwithinthresh] = sigmaclip2tail(sigmaY(sigmaY>0),numberofsigma);

figmau
hist(sigmaY,200)
xlabel('Sigma Y (pixels)')
hold on
plot([sigmaY_calc_lowthreshold sigmaY_calc_lowthreshold], [0 500],'r')
plot([sigmaY_calc_highthreshold sigmaY_calc_highthreshold], [0 500],'r')
hold off
title(['Threshold = [',num2str(sigmaY_calc_lowthreshold), ...
    ' ',num2str(sigmaY_calc_highthreshold),']'])
fig2pretty




%% Remove multi-emitters with Sigma_X, Sigma_Y and Peak_Ellipticity 

% Scatter sigmaX sigmaY with color as local density
vector1 = sigmaY;
vector2 = sigmaX;

vector1 = vector1(vector1>0);
vector2 = vector2(vector2>0);

numbins = 50;

binsfory = linspace(min(vector1),max(vector1),numbins);
binsforx = linspace(min(vector2),max(vector2),numbins);
h1506 = hist3([vector1 vector2],'Edges', {binsfory, binsforx} );

% % % % figmau
% % % % imagesc(h1506)
% % % % xlabel('Sigma X (nm)')
% % % % ylabel('Sigma Y (nm)')
% % % % axis image xy
% % % % fig2pretty
% % % % grid off

minvector1 = min(vector1);
minvector2 = min(vector2);
maxvector1 = max(vector1);
maxvector2 = max(vector2);
localdensity = zeros(size(vector1));
% Find local density from the hist3 data
parfor i = 1:length(vector1)
%     localdensity(i) = h1506( round(vector1(i)/max(vector1)*numbins), ...
%         round( vector2(i)/max(vector2)*numbins) );
    ybin = round((vector1(i)-minvector1) / (maxvector1-minvector1) *numbins);
    xbin = round((vector2(i)-minvector2) / (maxvector2-minvector2) *numbins);
    ybin(ybin==0) = 1;
    xbin(xbin==0) = 1;
    localdensity(i) = h1506(ybin, xbin);
end

%% Scatter plot but use the colors of the hist3 plot
figmau
scatter(vector2, vector1, 10, localdensity ,'filled')
xlabel('Sigma X (pixels)')
ylabel('Sigma Y (pixels)')
title(['Sigma X = [',num2str(sigmaX_calc_lowthreshold,2),' ',num2str(sigmaX_calc_highthreshold,2), ...
    ']. Sigma Y = [',num2str(sigmaY_calc_lowthreshold,2),' ',num2str(sigmaY_calc_highthreshold,2), ...
    ']. PE = [-',num2str(ellipticity_threshold),' +',num2str(ellipticity_threshold),']'])
axis image xy
c = colorbar;
c.Label.String = 'Arbitrary Local Density';
fig2pretty

% sigmax_threshold = [0.8 1.05];
% sigmay_threshold = [0.8 1.05];

hold on

plot([sigmaX_calc_lowthreshold sigmaX_calc_lowthreshold],[min(vector1) max(vector1)],'r')
plot([sigmaX_calc_highthreshold sigmaX_calc_highthreshold],[min(vector1) max(vector1)],'r')
plot([min(vector2) max(vector2)],[sigmaY_calc_lowthreshold sigmaY_calc_lowthreshold],'r')
plot([min(vector2) max(vector2)],[sigmaY_calc_highthreshold sigmaY_calc_highthreshold],'r')

xvector1273 = binsforx;
yvector1316 = xvector1273*(1+ellipticity_threshold/2)/(1-ellipticity_threshold/2);
yvector1317 = xvector1273*(1-ellipticity_threshold/2)/(1+ellipticity_threshold/2);
plot(xvector1273,yvector1316,'r')
plot(xvector1273,yvector1317,'r')
hold off

xlim([min(vector2) max(vector2)])
ylim([min(vector1) max(vector1)])
caxis([min(min(h1506(3:end-2,:))) max(max(h1506(3:end-2,:)))])
fig2pretty
grid off



%% Place three thresholds on each of the 1D histogram for Sigma_X, Sigma_Y and Peak_Ellipticity 
OK_result_sigmaX = true(size(meanresidual));
OK_result_sigmaY = true(size(meanresidual));
OK_result_PE = true(size(meanresidual));
OK_result_sigmaX(sigmaX < sigmaX_calc_lowthreshold) = 0;
OK_result_sigmaX(sigmaX > sigmaX_calc_highthreshold) = 0;
OK_result_sigmaY(sigmaY < sigmaY_calc_lowthreshold) = 0;
OK_result_sigmaY(sigmaY > sigmaY_calc_highthreshold) = 0;
OK_result_PE(Peak_Ellipticity < -ellipticity_threshold) = 0;
OK_result_PE(Peak_Ellipticity > ellipticity_threshold) = 0;

fprintf(['After placing thresholds with sigmaX, #not OK localizations = ',num2str(sum(OK_result_sigmaX==0)),'\n'])
fprintf(['After placing thresholds with sigmaY, #not OK localizations = ',num2str(sum(OK_result_sigmaY==0)),'\n'])
fprintf(['After placing thresholds with peak ellipticity, #not OK localizations = ',num2str(sum(OK_result_PE==0)),'\n'])
fprintf(fileID,['After placing thresholds with sigmaX, #not OK localizations = ',num2str(sum(OK_result_sigmaX==0)),'\r\n']);
fprintf(fileID,['After placing thresholds with sigmaY, #not OK localizations = ',num2str(sum(OK_result_sigmaY==0)),'\r\n']);
fprintf(fileID,['After placing thresholds with peak ellipticity, #not OK localizations = ',num2str(sum(OK_result_PE==0)),'\r\n']);

%% Use the three thresholds for sigma x, sigma y, and peak ellipticity before using the other thresholds
OK_result = and(OK_result, OK_result_sigmaX);
OK_result = and(OK_result, OK_result_sigmaY); 
OK_result = and(OK_result, OK_result_PE);


%% Look at mean residual
meanresidual_OK = meanresidual(OK_result);

num_sigma = 3;
[meanresidualthreshold,~] = sigmaclip(meanresidual_OK,num_sigma);

figmau
hist(meanresidual_OK,0:1:max(meanresidual_OK))
hold on
plot([min(meanresidual_OK) min(meanresidual_OK)],[0 1000],'r')
plot([median(meanresidual_OK) median(meanresidual_OK)],[0 1000],'r')
hold off
xlabel('Mean residual (photons/pixel)')
fig2pretty

hold on
plot([meanresidualthreshold meanresidualthreshold],[0 1000],'g')
hold off
title(['Mean residual threshold = ',num2str(meanresidualthreshold,3)])
% xlim([0 100])
fig2pretty

%% Look at localizations for mean residual
thisvalue = meanresidualthreshold;
thisvector = meanresidual_OK;
[~,thisrow] = min(abs(thisvector - thisvalue ));
[what,~] = find(meanresidual==thisvector(thisrow));
imagescmau(ROItofit(:,:,what(1)))

title(['Mean residual = ',num2str(thisvector(thisrow),3)])
fig2pretty
grid off

%% Apply the threshold to a vector specifically for this threshold
OK_result_meanresidual = true(size(meanresidual));
OK_result_meanresidual(meanresidual>meanresidualthreshold) = 0;
fprintf(['After placing thresholds for mean residual, #not OK localizations = ',num2str(sum(OK_result_meanresidual==0)),'\n'])
fprintf(fileID,['After placing thresholds for mean residual, #not OK localizations = ',num2str(sum(OK_result_meanresidual==0)),'\r\n']);


%% Look at max residual
maxresidual_OK = maxresidual(OK_result);

num_sigma = 3;
[maxresidualthreshold,~] = sigmaclip(maxresidual_OK,num_sigma);

figmau
hist(maxresidual_OK,0:2:max(maxresidual_OK))
hold on
plot([min(maxresidual_OK) min(maxresidual_OK)],[0 1000],'r')
plot([median(maxresidual_OK) median(maxresidual_OK)],[0 1000],'r')
% plot([2*median(maxresidual_OK)-min(maxresidual_OK) 2*median(maxresidual_OK)-min(maxresidual_OK)],[0 1000],'r')
hold off
xlabel('Max residual (photons/pixel)')
fig2pretty

% [meanresidualthreshold,~] = ginput(1);
% maxresidualthreshold = 2*median(maxresidual_OK)-min(maxresidual_OK);

hold on
plot([maxresidualthreshold maxresidualthreshold],[0 1000],'g')
hold off
title(['Max residual threshold = ',num2str(maxresidualthreshold,3)])
xlim([0 2000])
fig2pretty

%% Look at localizations for max residual
thisvalue = maxresidualthreshold;
thisvector = maxresidual_OK;
[~,thisrow] = min(abs(thisvector - thisvalue ));
[what,~] = find(maxresidual==thisvector(thisrow));
imagescmau(ROItofit(:,:,what(1)))
title(['Max residual = ',num2str(thisvector(thisrow),3)])
fig2pretty
grid off

%% Apply the threshold to a vector specifically for this threshold
OK_result_maxresidual = true(size(meanresidual));
OK_result_maxresidual(maxresidual>maxresidualthreshold) = 0;
fprintf(['After placing thresholds for max residual, #OK localizations = ',num2str(sum(OK_result_maxresidual==0)),'\n'])
fprintf(fileID,['After placing thresholds for max residual, #OK localizations = ',num2str(sum(OK_result_maxresidual==0)),'\r\n']);


%% Use the two thresholds for mean and max residual before placing the other later thresholds
OK_result = and(OK_result, OK_result_maxresidual);
OK_result = and(OK_result, OK_result_meanresidual);





%% Calculate Mortensen precision % multiply by 2.35 for resolution
% sigmaXY = sqrt((sigmaX.*(pixelsizenm/1E9)).^2+(sigmaY.*(pixelsizenm/1E9)).^2);
sigmaXY =  (sigmaX+sigmaY)./2.*(pixelsizenm/1E9); % convert Sigma to meters
bg_and_offset = background+offsetvec;
sigma_a_sq = (sigmaXY.^2+(pixelsizenm/1E9)^2/12); % meters

var_precision = sigma_a_sq./totalphotons.*(16/9 + 8*pi*sigma_a_sq.*bg_and_offset./totalphotons./(pixelsizenm/1E9)^2);
mortensenprecisionnm = sqrt(var_precision)*1E9;

%% Mortensen precision threshold
mortensenprecisionnm_OK = mortensenprecisionnm(OK_result);
figmau
hist(mortensenprecisionnm_OK(mortensenprecisionnm_OK<200),4000)
xlabel('Mortensen precision (nm)')

mortensenthreshold = 14;

hold on

plot([min(mortensenprecisionnm_OK) min(mortensenprecisionnm_OK)],[0 500],'r')

plot([median(mortensenprecisionnm_OK) median(mortensenprecisionnm_OK)],[0 500],'r')
plot([mortensenthreshold mortensenthreshold],[0 500],'g')
hold off
title(['Mortensen precision threshold = ',num2str(mortensenthreshold), ' nm'])
xlim([0 30])

%% Apply the threshold to a vector specifically for this threshold
OK_result_mort = true(size(meanresidual));
OK_result_mort(mortensenprecisionnm>mortensenthreshold) = 0;
fprintf(['After placing threshold for Mortensen precision. #not OK localizations = ',num2str(sum(OK_result_mort==0)),'\n'])
fprintf(fileID,['After placing threshold for Mortensen precision. #not OK localizations = ',num2str(sum(OK_result_mort==0)),'\r\n\r\n']);

%% Look at localizations
thisvalue = mortensenthreshold;
thisvector = mortensenprecisionnm_OK;
[~,thisrow] = min(abs(thisvector - thisvalue ));
[what,~] = find(mortensenprecisionnm==thisvector(thisrow));
imagescmau(ROItofit(:,:,what))
title(['Mortensen precision (nm) = ',num2str(thisvector(thisrow),3)])
c = colorbar;
c.Label.String = 'Photons';
fig2pretty
grid off



%% Report certain metrics
 
OK_result = and(OK_result, OK_result_mort);

% num_of_localizations
fprintf(['\nNumber of starting localizations = ',num2str(num_of_localizations),'. localizations passed = ',num2str(sum(OK_result)/length(OK_result)*100,2),' percent.\n'])
% number of ok localizations
numberofokspots = sum(OK_result);
fprintf(['Number of OK localizations = ',num2str(numberofokspots),'\n'])
% Totalphotons
medianphotons = median(totalphotons(OK_result));
fprintf(['Median photons per loc = ',num2str(round(medianphotons)),'\n'])
% photons per pixel in bg in median filter
medianbg = median(background(OK_result)); % per pixel
fprintf(['Median estimated background (photons/pixel) = ',num2str(medianbg,3),'\n'])
% offset
medianoffset = median(offsetvec(OK_result)); % per pixel
fprintf(['Median offset (photons/pixel) = ',num2str(medianoffset,3),'\n'])
% Median of Mortensen precision
medianmortensen = median(mortensenprecisionnm(OK_result));
fprintf(['Median Mortensen precision (nm) = ',num2str(medianmortensen,3),'\n\n'])

fprintf('Here are the thresholds used\n\n')
fprintf(['Threshold for sigma X = [',num2str(sigmaX_calc_lowthreshold,3),' ',num2str(sigmaX_calc_highthreshold,3),']. localizations failed = ',num2str(sum(OK_result_sigmaX==0)),' \n'])
fprintf(['Threshold for sigma Y = [',num2str(sigmaY_calc_lowthreshold,3),' ',num2str(sigmaY_calc_highthreshold,3),']. localizations failed = ',num2str(sum(OK_result_sigmaY==0)),' \n'])
fprintf(['Threshold for peak ellipticity = ',num2str(ellipticity_threshold),'. localizations failed = ',num2str(sum(OK_result_PE==0)),' \n\n'])

fprintf(['Threshold for mean residual/pixel (sum abs) = ',num2str(round(meanresidualthreshold)),'. localizations failed = ',num2str(sum(OK_result_meanresidual==0)),' \n'])
fprintf(['Threshold for max residual = ',num2str(round(maxresidualthreshold)),'. localizations failed = ',num2str(sum(OK_result_maxresidual==0)),' \n'])
fprintf(['Threshold for Mortensen precision (nm) = ',num2str(mortensenthreshold,3),'. localizations failed = ',num2str(sum(OK_result_mort==0)),' \n\n'])



fprintf(fileID,['\nNumber of starting localizations = ',num2str(num_of_localizations),'. localizations passed = ',num2str(sum(OK_result)/length(OK_result)*100,2),' percent.\r\n']);
fprintf(fileID,['Number of OK localizations = ',num2str(numberofokspots),'\r\n']);
fprintf(fileID,['Median photons per loc = ',num2str(round(medianphotons)),'\r\n']);
fprintf(fileID,['Median estimated background (photons/pixel) = ',num2str(medianbg,3),'\r\n']);
fprintf(fileID,['Median offset (photons/pixel) = ',num2str(medianoffset,3),'\r\n']);
fprintf(fileID,['Median Mortensen precision (nm) = ',num2str(medianmortensen,3),'\r\n\r\n']);
fprintf(fileID,'Here are the thresholds used\r\n\r\n');
fprintf(fileID,['Threshold for sigma X = [',num2str(sigmaX_calc_lowthreshold,3),' ',num2str(sigmaX_calc_highthreshold,3),']. localizations failed = ',num2str(sum(OK_result_sigmaX==0)),' \r\n']);
fprintf(fileID,['Threshold for sigma Y = [',num2str(sigmaY_calc_lowthreshold,3),' ',num2str(sigmaY_calc_highthreshold,3),']. localizations failed = ',num2str(sum(OK_result_sigmaY==0)),' \r\n']);
fprintf(fileID,['Threshold for peak ellipticity = ',num2str(ellipticity_threshold),'. localizations failed = ',num2str(sum(OK_result_PE==0)),' \r\n\r\n']);
fprintf(fileID,['Threshold for mean residual/pixel (sum abs) = ',num2str(round(meanresidualthreshold)),'. localizations failed = ',num2str(sum(OK_result_meanresidual==0)),' \r\n']);
fprintf(fileID,['Threshold for max residual = ',num2str(round(maxresidualthreshold)),'. localizations failed = ',num2str(sum(OK_result_maxresidual==0)),' \r\n']);
fprintf(fileID,['Threshold for Mortensen precision (nm) = ',num2str(mortensenthreshold,3),'. localizations failed = ',num2str(sum(OK_result_mort==0)),' \r\n\r\n']);


%% Final image scatter
% % % % % % % % figmau
% % % % % % % % scatter(Xnm78(OK_result) , Ynm78(OK_result) ,'k.')
% % % % % % % % hold off
% % % % % % % % xlabel('X (nm)')
% % % % % % % % ylabel('Y (nm)')
% % % % % % % % title('Scatter plot of chosen fitted localizations')
% % % % % % % % axis image xy
% % % % % % % % fig2pretty

%% Final image hist
% % % % % % % % sizeofbin = 50;
% % % % % % % % binsforx = min(Xnm78(OK_result)):sizeofbin:max(Xnm78(OK_result));
% % % % % % % % binsfory = min(Ynm78(OK_result)):sizeofbin:max(Ynm78(OK_result));
% % % % % % % % h29 = hist3([Ynm78(OK_result) Xnm78(OK_result)],{binsfory binsforx});
% % % % % % % % imagescmau(h29)
% % % % % % % % axis image xy
% % % % % % % % title(['1 bin is ',num2str(sizeofbin),' nm'])
% % % % % % % % % colormap(dusk)
% % % % % % % % colormap(inferno)
% % % % % % % % caxis([0 50])
% % % % % % % % fig2pretty









%% Phase 2
%% CHECK BLINKING %% CHECK BLINKING %% CHECK BLINKING 
% Look at the spots and calculate how far is the nearest spot within a
% distance threshold. measure both time and space
% use OK_result(ind) == 1

DistanceThreshold = 50; 
DistanceToNextSpotWithinThreshold = zeros(num_of_localizations,1);
FramesToNextSpotWithinThreshold = zeros(num_of_localizations,1);
whichframethisis = zeros(num_of_localizations,1);
for ind = 1:num_of_localizations
    thisX = X_nm(ind);
    thisY = Y_nm(ind);
    thisFrame = FrameNumber(ind);
    dist_to_other_X = X_nm - thisX;
    dist_to_other_Y = Y_nm - thisY;
    dist_to_other_XY = sqrt(dist_to_other_X.^2 + dist_to_other_Y.^2);
    dist_to_other_XY(dist_to_other_XY>DistanceThreshold) = NaN;
    dist_to_other_XY(FrameNumber<=thisFrame) = NaN; % remove all spots from previous frames
    Ind_Next_Nearest_Spot = find(dist_to_other_XY>0);
    if size(Ind_Next_Nearest_Spot,1)>0
        DistanceToNextSpotWithinThreshold(ind) = dist_to_other_XY(Ind_Next_Nearest_Spot(1));
        FramesToNextSpotWithinThreshold(ind) = FrameNumber(Ind_Next_Nearest_Spot(1)) - thisFrame;
        whichframethisis(ind) = thisFrame;
    else
        DistanceToNextSpotWithinThreshold(ind) = NaN;
        FramesToNextSpotWithinThreshold(ind) = NaN;
        whichframethisis(ind) = NaN;
    end
end

%%
densityforplot1300 = zeros(size(DistanceToNextSpotWithinThreshold));
for ii = 1:length(densityforplot1300)
    thisDist = DistanceToNextSpotWithinThreshold(ii);
    thisFrames = FramesToNextSpotWithinThreshold(ii);
    differenceinDist = DistanceToNextSpotWithinThreshold - thisDist;
    differenceinDist(abs(differenceinDist)>2) = NaN;
    differenceinFrame = FramesToNextSpotWithinThreshold - thisFrames;
    differenceinFrame(abs(differenceinFrame)>10) = NaN;
    
    differenceinDist(abs(differenceinFrame)>10) = NaN;
    differenceinFrame(abs(differenceinDist)>2) = NaN;
    
    densityforplot1300(ii) = numel(find(differenceinDist>0));
end
%%
figmau
scatter(DistanceToNextSpotWithinThreshold , FramesToNextSpotWithinThreshold,10,densityforplot1300,'filled','MarkerEdgeColor','k')
xlabel('Distance to next spot (nm)')
ylabel('Frames to next nearby spot')
ylim([0 200])
fig2pretty

%%
figmau
hist(FramesToNextSpotWithinThreshold(FramesToNextSpotWithinThreshold<100),0:100)
xlabel('#Frames to next spot within 50 nm')
fig2pretty
xlim([0 100])

%%
figmau
hist(DistanceToNextSpotWithinThreshold(DistanceToNextSpotWithinThreshold<=50),0:50)
xlabel('Distance to next spot within 50 nm')
fig2pretty
xlim([0 50])




%% HISTOGRAM number of localizations per frame
spotsperframe = zeros(max(FrameNumber),1,'single');
for thisframe = 1:max(FrameNumber)
    spotsperframe(thisframe) = numel(find(FrameNumber==thisframe));
end
figmau
hist(spotsperframe,1:max(spotsperframe))
xlabel('Number of localizations per frame')
fig2pretty

%% Plot number of localizations per frame
figmau
frame_vec = 1:1:max(FrameNumber);
plot(frame_vec, spotsperframe,'k.')
xlim([min(frame_vec) max(frame_vec)])
ylabel('Number of localizations per frame')
xlabel('Frame')
fig2pretty

%% Need to choose a distance cutoff to the nearest localization in the next frame to decide whether they are the same molecule

% IMPORTANT NUMBER, upper distance limit for calculating closeby molecules in subsequent frames
dist_cutoff = 60; % nm 

samespot_INDEX = zeros(num_of_localizations,1,'single'); % this is the index for the same molecule
samespot_distapart = zeros(num_of_localizations,totalnumberoffframes+1,'single'); % store the distance between the 2 localizations
num_molecules_left = zeros(size(samespot_distapart,2),1);

fprintf(['Calculating distance from a localization to nearest localization in next frame within ',num2str(dist_cutoff),' nm.\n'])
fprintf(fileID,['Calculating distance from a localization to nearest localization in next frame within ',num2str(dist_cutoff),' nm.\r\n']);

%% find the distance to the nearest localization in the next frame

for jj = 1:size(samespot_distapart,2)
    for ind = 1:num_of_localizations
        % check if this localization is ok
        if OK_result(ind) == 1
            % check if this localization has already appeared
            if samespot_INDEX(ind)==0
                % new localization! give it a new number!
                samespot_INDEX(ind) = max(samespot_INDEX(:))+1;
            end
            
            % check the next frame until the max number of off frames
            
            IND_innextframe = find(FrameNumber == FrameNumber(ind) + jj); % indices for all localizations in subsequent frame to be checked
            dist_between_spots = pdist2([X_nm(ind) Y_nm(ind)], ...
                [X_nm(IND_innextframe) Y_nm(IND_innextframe)]); % distance between this localization and all other localizations in frame to be checked
            isthereaspot = find( dist_between_spots < dist_cutoff); % check if there's a closeby localization
            
            % if there is a nearest localization in the next frame
            if isempty(isthereaspot) == 0
                isthereaspot = isthereaspot(1); % usually there is only one closeby localization
                if OK_result(IND_innextframe(isthereaspot)) == 1
                    samespot_INDEX(IND_innextframe(isthereaspot)) = samespot_INDEX(ind); % change the index of the next localization
                    samespot_distapart(IND_innextframe(isthereaspot),jj) = min(dist_between_spots); % calculate the distance between the two localizations
                end
            end
        end
    end
    num_molecules_left(jj) = numel(unique(samespot_INDEX))-1;
end

%% find the correct number of off frames
% get 99.9% of molecules
% plot number of molecules left with each number of off frame
figmau
plot(0:size(samespot_distapart,2)-1 , num_molecules_left)
xlabel('Number of off frames')
ylabel('#molecules left after accounting for off frames')
fig2pretty

diff_num_molecules_left = diff(num_molecules_left);
figmau
plot(1:size(samespot_distapart,2)-1 , diff_num_molecules_left)
xlabel('Number of off frames')
ylabel('Difference in #molecules left')
fig2pretty

diff_num_molecules_left_ratio = diff(num_molecules_left)./num_molecules_left(1:end-1);
figmau
plot(1:size(samespot_distapart,2)-1 , abs(diff_num_molecules_left_ratio)*100)
xlabel('Number of off frames')
ylabel('Percentage drop in #molecules left')
fig2pretty

%% HISTOGRAM number of localizations for each molecule (1/2)
num_MOLECULES = max(samespot_INDEX(:));
fprintf([num2str(sum(OK_result)),' OK localizations (out of ',num2str(num_of_localizations),') are merged into ',num2str(num_MOLECULES), ...
    ' molecules within ',num2str(dist_cutoff),' nm and with ',num2str(totalnumberoffframes),' off frames. \n'])
fprintf(fileID,[num2str(sum(OK_result)),' OK localizations (out of ',num2str(num_of_localizations),') are merged into ',num2str(num_MOLECULES), ...
    ' molecules within ',num2str(dist_cutoff),' nm and with ',num2str(totalnumberoffframes),' off frames. \r\n']);

output_matrix(:,1) = samespot_INDEX; % the first column of output_matrix is the molecule index

%% HISTOGRAM number of localizations for each molecule (2/2)
numberoflocforeachmolecule_log = zeros(num_MOLECULES,1,'single');

for ii = 1:num_MOLECULES
    numberoflocforeachmolecule_log(ii) = sum(samespot_INDEX==ii);
end

figmau
hist(numberoflocforeachmolecule_log,0:1:max(numberoflocforeachmolecule_log))
xlim([0 max(numberoflocforeachmolecule_log)])
ylabel('# molecules with #localizations')
xlabel('Number of localizations for each molecule')
fig2pretty

%% Calculate the distance from one localization to the localization in the next frame
% HISTOGRAM the distance between localizations and Find the threshold for distance between localizations
histmau(samespot_distapart(samespot_distapart>0))
num_sigma = 2.3;
[distancebtwloc_threshold,~] = sigmaclip(samespot_distapart(samespot_distapart>0),num_sigma);
% distancebtwloc_threshold = median(samespot_distapart(samespot_distapart>0)) + 2*std(samespot_distapart(samespot_distapart>0));
hold on
plot([min(samespot_distapart(samespot_distapart>0)) min(samespot_distapart(samespot_distapart>0))],[0 100],'r')
plot([median(samespot_distapart(samespot_distapart>0)) median(samespot_distapart(samespot_distapart>0))],[0 100],'r')
plot([distancebtwloc_threshold distancebtwloc_threshold],[0 100],'g')
hold off

xlabel('Distance between reappearing localizations (nm)')
ylabel('Number of localizations')
title(['Median=',num2str(median(samespot_distapart(samespot_distapart>0)),3),'nm. Threshold=',num2str(distancebtwloc_threshold,3),' nm'])
fig2pretty

fprintf(['Median distance between localizations = ',num2str(median(samespot_distapart(samespot_distapart>0)),3),' nm','\n'])
fprintf(['Distance threshold for two localizations to be same molecule = ',num2str(distancebtwloc_threshold,3),' nm','\n\n'])
fprintf(fileID,['Median distance between localizations = ',num2str(median(samespot_distapart(samespot_distapart>0)),3),' nm','\r\n']);
fprintf(fileID,['Distance threshold for two localizations to be same molecule = ',num2str(distancebtwloc_threshold,3),' nm','\r\n\r\n']);

%% Place threshold for distance apart between localizations
OK_result_distapart = true(size(meanresidual));
for jj = 1:size(samespot_distapart,2)
    OK_result_distapart(samespot_distapart(:,jj)> distancebtwloc_threshold) = 0;
end
% localizations that appeared once remain as TRUE in OK_result_distapart
% localizations that are within the distancebtwloc_threshold remain as TRUE
% in OK_result_distapart
% Don't analyze localizations that are within distancebtwloc_threshold (54nm) and dist_cutofff (80nm)

fprintf(['Threshold for distance between localizations = ',num2str(distancebtwloc_threshold,3),' nm.\n'])
fprintf(['After placing threshold for distance between localizations in different frames. #OK localizations = ',num2str(sum(OK_result_distapart)),'\n'])
fprintf(['After placing threshold for distance between localizations in different frames. #removed localizations = ',num2str(length(OK_result_distapart)-sum(OK_result_distapart)),'\n'])
fprintf(fileID,['Threshold for distance between localizations = ',num2str(distancebtwloc_threshold,3),' nm\r\n']);
fprintf(fileID,['After placing threshold for distance between localizations in different frames. #OK localizations = ',num2str(sum(OK_result_distapart)),'\r\n']);
fprintf(fileID,['After placing threshold for distance between localizations in different frames. #removed localizations = ',num2str(length(OK_result_distapart)-sum(OK_result_distapart)),'\r\n']);

%% Look at the 2 ROIs for each distance between localizations
% % thisvalue = 10.1;
% % thisvector = alldistapart;
% % [~,whichrow] = min(abs(thisvector-thisvalue));
% % whichrows = find(samespot_IND==samespot_IND(whichrow));
% % figure(1965)
% % subplot(1,2,1)
% % imagesc(ROItofit(:,:,whichrows(1)))
% % axis image xy
% % xlabel(['Frame ',num2str(framenumber(whichrows(1)))])
% % colorbar
% % fig2pretty
% % grid off
% % subplot(1,2,2)
% % imagesc(ROItofit(:,:,whichrow))
% % xlabel(['Frame ',num2str(framenumber(whichrow))])
% % axis image xy
% % 
% % thistwospotsdist = sqrt((Xnm(whichrows(1)) - Xnm(whichrow))^2 + ... 
% % (Ynm(whichrows(1)) - Ynm(whichrow))^2)*pixelsizenm;
% % title(['Dist. btw localizations (nm) = ',num2str(thistwospotsdist,3)])
% % colorbar
% % fig2pretty
% % grid off
% % c = colorbar;
% % c.Label.String = 'Photons';




%% Look at histogram of number of off frames
% numoff_frames_vec = zeros(length(OK_result),1);
% maxnum_off_frames = 10;
% for ind = 1:length(OK_result)
%     % check the next few frames and see if there are any localizations closeby    % within distancebtwloc_threshold
%     thisframe = framenumber(ind);
%     IND_innextframe = find(framenumber>thisframe & framenumber<=thisframe+maxnum_off_frames);
%     % check next nearest localization
%     dist_between_spots = pdist2([Xnm78(ind) Ynm78(ind)], [Xnm78(IND_innextframe) Ynm78(IND_innextframe)]);
%     isthereaspot = find( dist_between_spots < distancebtwloc_threshold);
%     nextframes_withnearbyspots = framenumber(IND_innextframe(isthereaspot));
%     % only if there are nearby localizations & only if the difference between
%     % thisframe and the next closest frame is more than 1 frame (i.e. at
%     % least one off frame)
%     if numel(nextframes_withnearbyspots) > 0 && min(nextframes_withnearbyspots)-thisframe>1
%         numoff_frames_vec(ind) = min(nextframes_withnearbyspots) - thisframe - 1;
%     end
% end
% %% Histogram number of off frames
% figure
% hist(numoff_frames_vec,0:maxnum_off_frames)
% xlabel('#off frames')
% fig2pretty












%% Calculate difference between each PSF
sum_abs_diff_ROI_log = zeros(length(OK_result),1,'single');

%% calculations
for ii = 1:num_MOLECULES % for each molecule
    whichrows = find(samespot_INDEX==ii); % find the rows in samespot_IND that belong to the same molecule
    if numberoflocforeachmolecule_log(ii) > 1 % if the molecule is turned on for more than 1 frame
        
        % Crop the ROIs based on the location of the first appearance of
        % the molecule
        thisx1 = output_matrix(whichrows(1),2)/pixelsizenm+0.5;
        thisy1 = output_matrix(whichrows(1),3)/pixelsizenm+0.5;
        
        %% Place all the ROIs from this molecule into a small stack
        thisframes = output_matrix(whichrows(:),12);
        thisROIs = DATA_wobg(round(thisy1)-ROIhalfwidth:round(thisy1)+ROIhalfwidth, ...
            round(thisx1)-ROIhalfwidth:round(thisx1)+ROIhalfwidth, thisframes);
        thisROIs = thisROIs./repmat(sum(sum(abs(thisROIs))), [ROIsize ROIsize 1]);
        
        %%
        diff_thisROIs = thisROIs - repmat(thisROIs(:,:,1),[1 1 numberoflocforeachmolecule_log(ii)]); % subtract the first ROI from each frame
        sum_abs_diff_ROI_log(whichrows) = squeeze(sum(sum(abs(diff_thisROIs))));
    end
end

% imagescmau(thisROIs(:,:,2),thisROIs(:,:,2),thisROIs(:,:,4),thisROIs(:,:,10))


%% Find threshold for sum abs difference btw ROIs
histmau(sum_abs_diff_ROI_log(sum_abs_diff_ROI_log>0))

num_sigma = 2.3;
[sumabsdiffROIlog_threshold,~] = sigmaclip(sum_abs_diff_ROI_log(sum_abs_diff_ROI_log>0),num_sigma);
hold on
plot([min(sum_abs_diff_ROI_log(sum_abs_diff_ROI_log>0)) min(sum_abs_diff_ROI_log(sum_abs_diff_ROI_log>0))],[0 100],'r')
plot([median(sum_abs_diff_ROI_log(sum_abs_diff_ROI_log>0)) median(sum_abs_diff_ROI_log(sum_abs_diff_ROI_log>0))],[0 100],'r')
plot([sumabsdiffROIlog_threshold sumabsdiffROIlog_threshold],[0 100],'g')
hold off
xlabel('Sum abs difference between ROI')
title(['Sum abs diff thresh = ',num2str(sumabsdiffROIlog_threshold,3)])
fig2pretty

%% Look at the 2 ROIs for each value of sum abs diff ROI
thisvalue = sumabsdiffROIlog_threshold;
thisvector = sum_abs_diff_ROI_log;

[~,whichrow] = min(abs(thisvector-thisvalue));
whichrows = find(samespot_INDEX == samespot_INDEX(whichrow));
figmau

subplot(1,2,1)
imagesc(ROItofit(:,:,whichrows(1)))
axis image xy
colorbar
fig2pretty
grid off

subplot(1,2,2)
imagesc(ROItofit(:,:,whichrow(1)))
title(['Sum abs diff ROI = ',num2str(thisvector(whichrow),3)])
axis image xy
colorbar
fig2pretty
grid off



%% Remove fits that are close to each other but are still different
OK_result_diff_ROI = true(size(meanresidual));
OK_result_diff_ROI(sum_abs_diff_ROI_log> sumabsdiffROIlog_threshold) = 0;

fprintf(['Threshold for difference between ROIs [0 1] = ',num2str(sumabsdiffROIlog_threshold,3),' \n'])
fprintf(['After placing threshold for difference between ROIs. #OK localizations = ',num2str(sum(OK_result_diff_ROI)),'\n'])
fprintf(['After placing threshold for difference between ROIs. #removed localizations = ',num2str(sum(sum_abs_diff_ROI_log> sumabsdiffROIlog_threshold)),'\n'])
fprintf(fileID,['Threshold for difference between ROIs [0 1] = ',num2str(sumabsdiffROIlog_threshold,3),' \r\n']);
fprintf(fileID,['After placing threshold for difference between ROIs. #OK localizations = ',num2str(sum(OK_result_diff_ROI)),'\r\n']);
fprintf(fileID,['After placing threshold for difference between ROIs. #removed localizations = ',num2str(sum(sum_abs_diff_ROI_log> sumabsdiffROIlog_threshold)),'\r\n']);




%%











%% After omitting localizations that were too close to each other
OK_result = and(OK_result, OK_result_distapart); 
OK_result = and(OK_result, OK_result_diff_ROI);

output_matrix(:,14) = OK_result; % these localizations passed through ALL the filters after the first Gaussian fit

% number of ok localizations
numberofokspots = sum(OK_result);
fprintf(['After omitting localizations that appeared multiple times and were too far or too different. #OK localizations = ',num2str(numberofokspots),'\n']);
fprintf(fileID,['After omitting localizations that were appeared multiple times and were too far or too different. #OK localizations = ',num2str(numberofokspots),'\r\n']);

%% Find localizations that are actually the same molecule
% Calculate samespot_IND again
samespot_INDEX2 = zeros(length(OK_result),1,'single');

% Find the distance to the nearest localization in the next frame
for ind = 1:length(OK_result)
    % check if this localization is ok
    if OK_result(ind) == 1
        % check if this localization has already appeared
        if samespot_INDEX2(ind)==0
            % new localization! give it a new number!
            samespot_INDEX2(ind) = max(samespot_INDEX2(:))+1;
        end
        
        % check the next frame until the max number of off frames
        for jj = 1:totalnumberoffframes+1
            IND_innextframe = find(FrameNumber == FrameNumber(ind) + jj); % indices for all localizations in subsequent frame to be checked
            dist_between_spots = pdist2([X_nm(ind) Y_nm(ind)], ...
                [X_nm(IND_innextframe) Y_nm(IND_innextframe)]); % distance between this localization and all other localizations in frame to be checked
            isthereaspot = find( dist_between_spots < dist_cutoff); % check if there's a closeby localization
            
            % if there is a nearest localization in the next frame
            if isempty(isthereaspot) == 0
                isthereaspot = isthereaspot(1); % usually there is only one closeby localization
                if OK_result(IND_innextframe(isthereaspot)) == 1
                    samespot_INDEX2(IND_innextframe(isthereaspot)) = samespot_INDEX2(ind); % change the index of the next localization
                end
            end
        end
    end
    
    
end

%% Fit assymetric Gaussian with offset as a variable
num_MOLECULES2 = max(samespot_INDEX2(:));
fprintf(['#molecules within ',num2str(dist_cutoff),'nm and are OK = ',num2str(num_MOLECULES2),'\n'])
fprintf(fileID,['#molecules within ',num2str(dist_cutoff),'nm and are OK = ',num2str(num_MOLECULES2),'\r\n']);

output_matrix2 = zeros(num_MOLECULES2,14,'single');

totalphotons2 = zeros(num_MOLECULES2,1,'single');
X_pixels2 = zeros(num_MOLECULES2,1,'single');
Y_pixels2 = zeros(num_MOLECULES2,1,'single');
sigmaX2 = zeros(num_MOLECULES2,1,'single');
sigmaY2 = zeros(num_MOLECULES2,1,'single');
thetavec2 = zeros(num_MOLECULES2,1,'single');
meanresidual2 = zeros(num_MOLECULES2,1,'single');
maxresidual2 = zeros(num_MOLECULES2,1,'single');
offset2 = zeros(num_MOLECULES2,1,'single');
framevec2 = zeros(num_MOLECULES2,1,'single');
framelengthvec = zeros(num_MOLECULES2,1,'single');
background2 = zeros(num_MOLECULES2,1,'single');

%% Form a new ROItofit
SumAllPixelsLog2 = zeros(num_MOLECULES2,1,'single');
min_ROItofitnowlog2 = zeros(num_MOLECULES2,1,'single');
newROItofit = zeros(ROIsize,ROIsize,num_MOLECULES2,'single');
thismoleculeINDlog = zeros(num_MOLECULES2,1,'single'); 

for loc_ID = 1:num_MOLECULES2
    %% Sum all the frames where the molecule is present
    INDICES_for_thismolecule = find(samespot_INDEX2==loc_ID);
    
    mid_x_round = round(WC_x(INDICES_for_thismolecule(1)));
    mid_y_round = round(WC_y(INDICES_for_thismolecule(1)));
    pixels_x = mid_x_round-ROIhalfwidth:mid_x_round+ROIhalfwidth;
    pixels_y = mid_y_round-ROIhalfwidth:mid_y_round+ROIhalfwidth;
    
    thisframes = FrameNumber(INDICES_for_thismolecule);
    
    newROItofit(:,:,loc_ID) = sum(DATA_wobg(pixels_y,pixels_x,thisframes),3);

%     newROItofit(:,:,loc_ID) = sum(ROItofit(:,:,INDICES_for_thismolecule),3);
    SumAllPixelsLog2(loc_ID) = sum(sum(newROItofit(:,:,loc_ID)));
    min_ROItofitnowlog2(loc_ID) = min(min(newROItofit(:,:,loc_ID)));
    frames_thismolecule = FrameNumber(INDICES_for_thismolecule);
    framevec2(loc_ID) =  frames_thismolecule(1); % first frame
    
    thismoleculeINDlog(loc_ID) = INDICES_for_thismolecule(1);
    
    background2(loc_ID) = ...
        sum(output_matrix( INDICES_for_thismolecule ,9)); % sum of background
    
    framelengthvec(loc_ID) = length(INDICES_for_thismolecule); % number of frames
    
end
DC_log2 = min_ROItofitnowlog2;

%%
fprintf(['We are going to fit ',num2str(num_MOLECULES2),' combined ROI with a 2D Gaussian.\n'])
fprintf(['This will take approx. ',num2str(round(num_MOLECULES2/174/60)),' mins with 4 cores.\n\n'])
fprintf(fileID,['We are going to fit ',num2str(num_MOLECULES2),' combined ROI with a 2D Gaussian.\r\n']);
fprintf(fileID,['This will take approx. ',num2str(round(num_MOLECULES2/174/60)),' mins with 4 cores.\r\n\r\n']);

sigmadelta = 0.05;
sigmaXlowerthreshold = sigmax_threshold(1)-sigmadelta;
sigmaYlowerthreshold = sigmay_threshold(1)-sigmadelta;
sigmaXhigherthreshold = sigmax_threshold(2)+sigmadelta;
sigmaYhigherthreshold = sigmay_threshold(2)+sigmadelta;
maxXpixelinROI = ROIsize-1;
maxYpixelinROI = ROIsize-1;
XYpixelinitialguess = ceil(ROIsize/2);
meansigmaXguess = mean(sigmax_threshold);
meansigmaYguess = mean(sigmay_threshold);
% Fit the summed ROIs
clear par fitparam residual
[ii,jj] = meshgrid(1:ROIsize,1:ROIsize);

parfor loc_ID = 1:num_MOLECULES2

    %% Obtain one frame to fit
%     loc_ID  = 800
    newROItofitnow = newROItofit(:,:,loc_ID);
%     imagescmau(newROItofitnow)

    %% Do the 2D Gaussian fit
    InitialIntensityGuess = SumAllPixelsLog2(loc_ID); % initial guess for amplitude
    min_ROItofitnow = min_ROItofitnowlog2(loc_ID); % initial guess for dimmest pixel
    
    % asymmetric Gaussian fit
    %     lowerBound = [500; 2; 2; sigmaXlowerthreshold; sigmaYlowerthreshold; thetalimlow; min_ROItofitnow];
    %     upperBound = [InitialIntensityGuess; maxXpixelinROI; maxYpixelinROI; ...
    %         sigmaXhigherthreshold; sigmaYhigherthreshold; thetalimhigh; InitialIntensityGuess];
    %     DC = DC_log2(loc_ID) + 30;
    
    % % % %     [fitparam,~,residual,~,~,~,~] = ...
    % % % %         lsqnonlin(@(par) singleIntegratedGaussianOffset( par,double(newROItofitnow),ii,jj), ...
    % % % %         double([0.6*InitialIntensityGuess; XYpixelinitialguess; XYpixelinitialguess; meansigmaXguess; meansigmaYguess; thetastart; DC]), ...
    % % % %         double(lowerBound), ...
    % % % %         double(upperBound),options);
    
    % symmetric Gaussian fit
    lowerBound = [500; 2; 2; sigmaXlowerthreshold; meansigmaYguess; thetastart; min_ROItofitnow];
    upperBound = [InitialIntensityGuess; maxXpixelinROI; maxYpixelinROI; ...
        sigmaXhigherthreshold; meansigmaYguess; thetastart; InitialIntensityGuess];
    DC = min_ROItofitnow + 30;
    [fitparam,~,residual,~,~,~,~] = ...
        lsqnonlin(@(par) singleIntegratedGaussianOffset_symmetricversion( par,double(newROItofitnow),ii,jj), ...
        double([0.6*InitialIntensityGuess; XYpixelinitialguess; XYpixelinitialguess; meansigmaXguess; meansigmaYguess; thetastart; DC]), ...
        double(lowerBound), ...
        double(upperBound),options);
    
    %   par(1) = intensity
    %   par(2) = Xo
    %   par(3) = Yo
    %   par(4) = sigmaX;
    %   par(5) = sigmaY;
    %   par(6) = theta; degrees
    %   par(7) = DC;
    
    sum_sum_abs_residual = sum(abs(residual(:)));
    
%% integrated PSF troubleshooting

% % %         figmau
% % %         subplot(2,2,1)
% % %         imagesc(newROItofitnow)
% % %         title(num2str(loc_ID))
% % %         colorbar
% % %         axis image xy
% % %         
% % %         subplot(2,2,2)
% % %         par = fitparam;
% % % % % % % %         theta = par(6)*pi/180;
% % % % % % % %         E_x = 0.5*( erf( ((ii-par(2))*cos(theta) + (jj-par(3))*sin(theta) + 0.5) / (2*par(4)*par(4))) - ...
% % % % % % % %             erf( ((ii-par(2))*cos(theta) + (jj-par(3))*sin(theta) - 0.5) / (2*par(4)*par(4))));
% % % % % % % %         E_y = 0.5 * ( erf( (-(ii-par(2))*sin(theta) + (jj-par(3))*cos(theta) + 0.5) / (2*par(5)*par(5))) - ...
% % % % % % % %             erf( (-(ii-par(2))*sin(theta) + (jj-par(3))*cos(theta) - 0.5) / (2*par(5)*par(5))));
% % % % % % % %         mu = par(1).*E_x.*E_y + par(7);
% % %         
% % %         % symmetric
% % %         E_x = 0.5 * ( erf( (ii-par(2)+0.5) / (2*par(4)*par(4)) ) - erf( (ii-par(2)-0.5) / (2*par(4)*par(4)) ));
% % %         E_y = 0.5 * ( erf( (jj-par(3)+0.5) / (2*par(4)*par(4)) ) - erf( (jj-par(3)-0.5) / (2*par(4)*par(4)) ));
% % %         mu = par(1).*E_x.*E_y + par(7);
% % %         imagesc(mu)
% % %         axis image xy
% % %         colorbar
% % %         
% % %         subplot(2,2,4)
% % %         imagesc(reshape(residual,size(mu)))
% % %         xlabel(['Sum of residual=',num2str(sum_sum_abs_residual )])
% % %         title(['theta = ',num2str(fitparam(6)/pi*180,3),' degrees'])
% % %         colorbar
% % %         axis image xy
        
    %% Outputs
    thismoleculeIND = thismoleculeINDlog(loc_ID);
    totalphotons2(loc_ID) = fitparam(1); % total photons
    X_pixels2(loc_ID) = fitparam(2)+ WC_x_round(thismoleculeIND) -ROIhalfwidth-1; % x location
    Y_pixels2(loc_ID) =  fitparam(3)+ WC_y_round(thismoleculeIND) -ROIhalfwidth-1; % y location
    sigmaX2(loc_ID) = fitparam(4); % sigma X
%     sigmaY2(loc_ID) = fitparam(5); % sigma Y
%     thetavec2(loc_ID) = fitparam(6); % theta, degrees
    meanresidual2(loc_ID) = sum_sum_abs_residual/ROIsize/ROIsize; % residual
    if size(residual,1) > 0
        maxresidual2(loc_ID) = max(abs(residual(:)));
    end
    offset2(loc_ID) = fitparam(7); % photons per pixel
    
end

%% Place the parallelized vectors into a table
X_nm2 = (X_pixels2-0.5)*pixelsizenm;
Y_nm2 = (Y_pixels2-0.5)*pixelsizenm;
% output_matrix2(:,1) is blank. the row number is the molecule ID
output_matrix2(:,2) = X_nm2; % x location
output_matrix2(:,3) = Y_nm2; % y location
output_matrix2(:,4) = sigmaX2; % sigma X, pixels
% output_matrix2(:,5) = sigmaY2; % sigma Y, pixels
output_matrix2(:,5) = sigmaX2; % sigma Y, pixels
sigmaY2 = sigmaX2;
output_matrix2(:,6) = thetavec2; % theta, degrees
output_matrix2(:,7) = totalphotons2; % total photons
output_matrix2(:,8) = offset2; % photons per pixel offset
output_matrix2(:,9) = background2; % sum of background
output_matrix2(:,10) = meanresidual2; % residual
output_matrix2(:,11) = maxresidual2; % max residual
output_matrix2(:,12) = framevec2; % first frame
output_matrix2(:,13) = framelengthvec; % number of frames

% 1 == molecule_ID
% 2 == X (nm)
% 3 == Y (nm)
% 4 == sigmaX
% 5 == sigmaY
% 6 == theta
% 7 == totalphotons
% 8 == offset
% 9 == meanbackground (from earlier estimation)
% 10 == mean residual
% 11 == max residual
% 12 == frame first appeared
% 13 == total number of frames
% 14 == OK_result2

%% Scatter plot of centroids
figmau
plot(output_matrix2(:,2), ...
    output_matrix2(:,3) ,'k.');
xlabel('X (nm)')
ylabel('Y (nm)')
axis image xy
fig2pretty

%% check for pixel locking
figmau
hist_pixellocking = hist3([mod(output_matrix2(:,3),pixelsizenm) mod(output_matrix2(:,2),pixelsizenm)], ...
    'Ctrs',{5:10:pixelsizenm-5 5:10:pixelsizenm-5});
imagesc(hist_pixellocking)
xlabel('Mod X (nm)')
ylabel('Mod Y(nm)')
title('Check for pixel locking using modulo of pixelsizenm')
axis image xy
colorbar
colormap(inferno)
set(gca,'YTick',[])
set(gca,'XTick',[])
fig2pretty
grid off

%% Change some of the measured variables into vectors
ind_ok = find(totalphotons2>0);

%% Final image hist after combining ROIs
sizeofbin = 50;
binsforx = min(X_nm2):sizeofbin:max(X_nm2);
binsfory = min(Y_nm2):sizeofbin:max(Y_nm2);
h39 = hist3([Y_nm2 X_nm2],{binsfory binsforx});
imagescmau(h39)
axis image xy
title(['1 bin is ',num2str(sizeofbin),' nm'])
caxis([0 30])
fig2pretty

%%
Peak_Ellipticity2 = 2*(sigmaX2-sigmaY2)./(sigmaX2+sigmaY2); % Peak ellipticity formula (XWZ nat meth 2006)
OK_result2 = true(num_MOLECULES2,1);




%% Remove multi-emitters with Sigma_X, Sigma_Y and Peak_Ellipticity 
% Scatter sigmaX sigmaY with color as local density
vector1 = sigmaY2;
vector2 = sigmaX2;

vector1 = vector1(vector1>0);
vector2 = vector2(vector2>0);

numbins = 50;

binsfory = linspace(min(vector1),max(vector1),numbins);
binsforx = linspace(min(vector2),max(vector2),numbins);
h1506 = hist3([vector1 vector2],'Edges', {binsfory, binsforx} );

% % % % figmau
% % % % imagesc(h1506)
% % % % xlabel('Sigma X (nm)')
% % % % ylabel('Sigma Y (nm)')
% % % % axis image xy
% % % % fig2pretty
% % % % grid off

minvector1 = min(vector1);
minvector2 = min(vector2);
maxvector1 = max(vector1);
maxvector2 = max(vector2);
localdensity = zeros(size(vector1));
% Find local density from the hist3 data
parfor i = 1:length(vector1)
%     localdensity(i) = h1506( round(vector1(i)/max(vector1)*numbins), ...
%         round( vector2(i)/max(vector2)*numbins) );
    ybin = round((vector1(i)-minvector1) / (maxvector1-minvector1) *numbins);
    xbin = round((vector2(i)-minvector2) / (maxvector2-minvector2) *numbins);
    ybin(ybin==0) = 1;
    xbin(xbin==0) = 1;
    localdensity(i) = h1506(ybin, xbin);
end


%% Scatter plot but use the colors of the hist3 plot
figmau
scatter(vector2, vector1, 10, localdensity ,'filled')
xlabel('Sigma X (pixels)')
ylabel('Sigma Y (pixels)')
title(['Sigma X = [',num2str(sigmaX_calc_lowthreshold,2),' ',num2str(sigmaX_calc_lowthreshold,2), ...
    ']. Sigma Y = [',num2str(sigmaY_calc_lowthreshold,2),' ',num2str(sigmaY_calc_highthreshold,2), ...
    ']. PE = [-',num2str(ellipticity_threshold),' +',num2str(ellipticity_threshold),']'])
axis image xy
c = colorbar;
c.Label.String = 'Arbitrary Local Density';
fig2pretty

% sigmax_threshold = [0.8 1.05];
% sigmay_threshold = [0.8 1.05];

hold on

plot([sigmaX_calc_lowthreshold sigmaX_calc_lowthreshold],[min(vector1) max(vector1)],'r')
plot([sigmaX_calc_highthreshold sigmaX_calc_highthreshold],[min(vector1) max(vector1)],'r')
plot([min(vector2) max(vector2)],[sigmaY_calc_lowthreshold sigmaY_calc_lowthreshold],'r')
plot([min(vector2) max(vector2)],[sigmaY_calc_highthreshold sigmaY_calc_highthreshold],'r')

xvector1273 = binsforx;
yvector1316 = xvector1273*(1+ellipticity_threshold/2)/(1-ellipticity_threshold/2);
yvector1317 = xvector1273*(1-ellipticity_threshold/2)/(1+ellipticity_threshold/2);
plot(xvector1273,yvector1316,'r')
plot(xvector1273,yvector1317,'r')
hold off

% xlim([min(vector2) max(vector2)])
% ylim([min(vector1) max(vector1)])
caxis([min(min(h1506(3:end-2,:))) max(max(h1506(3:end-2,:)))])
fig2pretty
grid off

%% Look at each localization in the scatter plot of sigma X vs sigma Y
% % % % xinput = 1.5;
% % % % yinput = 0.9;
% % % % 
% % % % D1270 = pdist2([xinput yinput],[Sigma_X Sigma_Y]);
% % % % [~,nearestmoleculeIND] = min(D1270);
% % % % whichrows = nearestmoleculeIND;
% % % % 
% % % % thisframe1 = photonsXYsigmaXYtheta_residuals_bg_offset_frames_combined(whichrows(1),10);
% % % % thisnumberframe1 = photonsXYsigmaXYtheta_residuals_bg_offset_frames_combined(whichrows(1),11);
% % % % thisx1 = photonsXYsigmaXYtheta_residuals_bg_offset_frames_combined(whichrows(1),2);
% % % % thisy1 = photonsXYsigmaXYtheta_residuals_bg_offset_frames_combined(whichrows(1),3);
% % % % 
% % % % figure
% % % % widthforthis = 3;
% % % % imagesc(newROItofit(:,:,nearestmoleculeIND))
% % % % axis image xy
% % % % hold on
% % % % plot(thisx1-round(thisx1)+widthforthis+1, ...
% % % %     thisy1-round(thisy1)+widthforthis+1, ...
% % % %     'rx','MarkerSize',20)
% % % % plot(thisx1-round(thisx1)+widthforthis+1, ...
% % % %     thisy1-round(thisy1)+widthforthis+1, ...
% % % %     'ko','MarkerSize',20)
% % % % 
% % % % title(['Frame=',num2str(thisframe1),' to ',num2str(1)])
% % % % xlabel(['X=',num2str(round(thisx1)),'. Y=',num2str(round(thisy1))])
% % % % hold off
% % % % c = colorbar;
% % % % c.Label.String = 'Photons';
% % % % fig2pretty

%% Place three thresholds on each of the 1D histogram for Sigma_X, Sigma_Y and Peak_Ellipticity 

figmau
hist(sigmaX2,200)
xlabel('Sigma X (pixels)')
hold on
plot([sigmaX_calc_lowthreshold sigmaX_calc_lowthreshold], [0 500],'r')
plot([sigmaX_calc_highthreshold sigmaX_calc_highthreshold], [0 500],'r')
hold off
title(['Threshold = [',num2str(sigmaX_calc_lowthreshold), ...
    ' ',num2str(sigmaX_calc_highthreshold),']'])
fig2pretty

OK_result2(sigmaX2 < sigmaX_calc_lowthreshold) = 0;
OK_result2(sigmaX2 > sigmaX_calc_highthreshold) = 0;


% sigma_Y (:,5)
figmau
hist(sigmaY2,200)
xlabel('Sigma Y (pixels)')
hold on
plot([sigmaY_calc_lowthreshold sigmaY_calc_lowthreshold], [0 500],'r')
plot([sigmaY_calc_highthreshold sigmaY_calc_highthreshold], [0 500],'r')
hold off
title(['Threshold = [',num2str(sigmaY_calc_lowthreshold), ...
    ' ',num2str(sigmaY_calc_highthreshold),']'])
fig2pretty

OK_result2(sigmaY2 < sigmaY_calc_lowthreshold) = 0;
OK_result2(sigmaY2 > sigmaY_calc_highthreshold) = 0;


% Peak_Ellipticity
figmau
hist(Peak_Ellipticity2 ,200)
xlabel('Peak Ellipticity = 2*(A-B)/(A+B)')
title(['Threshold = ',num2str(ellipticity_threshold)])
hold on
plot([ellipticity_threshold ellipticity_threshold], [0 500],'r')
plot([-ellipticity_threshold -ellipticity_threshold ], [0 500],'r')
hold off
fig2pretty

OK_result2(Peak_Ellipticity2 < -ellipticity_threshold) = 0;
OK_result2(Peak_Ellipticity2 > ellipticity_threshold) = 0;

fprintf(['#OK molecules 2 before removing fits = ',num2str(length(OK_result2)),'\n'])
fprintf(['Removed #fits = ',num2str(length(OK_result2)-sum(OK_result2)),'\n'])
fprintf(['#OK molecules 2 after removing fits = ',num2str(sum(OK_result2)),'\n'])

fprintf(fileID,['#OK molecules 2 before removing fits = ',num2str(length(OK_result2)),'\r\n']);
fprintf(fileID,['Removed #fits = ',num2str(length(OK_result2)-sum(OK_result2)),'\r\n']);
fprintf(fileID,['#OK molecules 2 after removing fits = ',num2str(sum(OK_result2)),'\r\n']);

%% Select only the rows that are OK
output_matrix2(:,14) = OK_result2;

%% Save the data. The 3 big tables and ROI to fit
% photonsXYsigmaXYtheta_residuals_bg_offset_frames
% photonsXYsigmaXYtheta_residuals_bg_offset_frames_combined
% photonsXYsigmaXYtheta_residuals_bg_offset_frames_combined_2
% ROItofit

%% Final image hist
sizeofbin = 40;
binsforx = min(X_nm2(OK_result2)):sizeofbin:max(X_nm2(OK_result2));
binsfory = min(Y_nm2(OK_result2)):sizeofbin:max(Y_nm2(OK_result2));
h2501 = hist3([Y_nm2(OK_result2) X_nm2(OK_result2)],{binsfory binsforx});

figmau
subplot(1,2,2)
imagesc(h2501)
title(['1 bin is ',num2str(sizeofbin),' nm'])
colorbar
colormap(inferno)
set(gca,'xticklabel',{[]}) 
set(gca,'yticklabel',{[]})
title('SR histogram')
fig2pretty
axis image xy
caxis([0 10])

subplot(1,2,1)
imagesc(sumofallframes)
axis image xy
% caxis([1.098E6 1.917E6])
set(gca,'xticklabel',{[]}) 
set(gca,'yticklabel',{[]})
title('Sum of all frames')
axis image xy
colorbar
fig2pretty



%% Save output_matrix and output_matrix2
filenameforoutputmatrix = [dataFile1(1:end-4) '_output_matrix1and2.mat'];
save(filenameforoutputmatrix,'output_matrix','output_matrix2')
movefile(filenameforoutputmatrix,pathtoanalysisfolder);

%% Blank space at end of code
fprintf('Done with analyzing dataset!\n')
ynt = clock;
fprintf(['Ending date and time: ', ...
    num2str(ynt(1)),num2str(ynt(2),'%02d'),num2str(ynt(3),'%02d'),' ', ...
    num2str(ynt(4)),':',num2str(ynt(5),'%02d'),' hrs\n'])

fprintf(fileID,'Done with analyzing dataset!\r\n');
fprintf(fileID,['Ending date and time: ', ...
    num2str(ynt(1)),num2str(ynt(2),'%02d'),num2str(ynt(3),'%02d'),' ', ...
    num2str(ynt(4)),':',num2str(ynt(5),'%02d'),' hrs\r\n']);

%% Same the entire command window as a text file

fclose('all');

copyfile(commandwindowlogfilename, pathtoanalysisfolder);

%% Play chirp when done
load chirp

for PeteDahlberg = 1:4
    sound(y,Fs)
    pause(1)
end









%% Look at ROItofit
% % % % % figure(789)
% % % % % for whichspot = 1:10:size(ROItofit,3)
% % % % %     if OK_result(whichspot) == 1
% % % % %         imagesc(ROItofit(:,:,whichspot))
% % % % %         xlabel(['Spot #',num2str(whichspot)])
% % % % %         colormap(inferno)
% % % % %         axis image xy
% % % % %         colorbar
% % % % %         fig2pretty
% % % % %         pause
% % % % %     end
% % % % % end














