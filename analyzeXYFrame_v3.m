% analyzeXYFrame_v3.m

% This script takes X Y Frame and analyzes the data

% v2 used VoronoiClustering.m to do Voronoi clustering
% v3 removed duplicates before doing any clustering.


%% Select file to load

figurecounter = 1;

load inferno.mat

% Import files for XY frame
[dataFile1, dataPath] = uigetfile({'*.mat';'*.*'},'Open .mat file with X Y frame data');
if isequal(dataFile1,0), error('User cancelled the program'); end

%% Get some information about the file

dataFile = [dataPath dataFile1];
disp(['File name = ', (dataFile)])
load(dataFile,'-mat')

%% Create a new analysis folder in the folder where the data was taken from
ynt = clock;
YMD = [num2str(ynt(1)) num2str(ynt(2),'%02d') num2str(ynt(3),'%02d') num2str(ynt(4),'%02d') num2str(ynt(5),'%02d')];
analysisFolderName = [YMD '_' dataFile1(1:end-4) '_Analysis'];
pathtoanalysisfolder = [dataPath analysisFolderName];
[status,~] = mkdir(pathtoanalysisfolder);

% Make the smallest X and Y value be 0
X = X1_corr - min(X1_corr);
Y = Y1_corr - min(Y1_corr);

%% REMOVE SPOTS (only if need be)

% for ii = 1:length(frame1)
%     if frame1(ii) > 558 && frame1(ii) < 619 || ...
%             frame1(ii) > 1363 && frame1(ii) < 1429 || ...
%             frame1(ii) > 2304 && frame1(ii) < 2371 || ...
%         frame1(ii) > 3255 && frame1(ii) < 3325
%     X(ii) = 0;
%     end
% end
% 
% frame1(X==0) =[];
% Y(X==0) = [];
% X(X==0) = [];

% frame1(X<2E3) =[];
% Y(X<2E3) = [];
% X(X<2E3) = [];

% for ii = 1:length(frame1)
%     if frame1(ii) > 1550 && frame1(ii) < 1607 
%     X(ii) = 0;
%     end
% end
% 
% frame1(X==0) =[];
% Y(X==0) = [];
% X(X==0) = [];

%% Make scatter plot with color representing frame number
figmau
scatter(X,Y,10,frame1,'filled')
xlabel('X (nm)')
ylabel('Y (nm)')
title('Scatter plot. Color is frame #')
axis image xy
fig2pretty
colormap(parula)
colorbar


%%
% % % %% Calculate appropriate radius to count number of neighbors
% % % radiustochoosefrom = [10 20 30 40 50 60 70]; % nm
% % % num_neighbors = zeros(length(X),length(radiustochoosefrom));
% % % 
% % % tic
% % % for molecule_ID = 1:length(X)
% % %     
% % %     thisX = X(molecule_ID);
% % %     thisY = Y(molecule_ID);
% % %     diff_thisX = X - thisX;
% % %     diff_thisY = Y - thisY;
% % %     diff_thisXY = sqrt(diff_thisX.^2+diff_thisY.^2);
% % %     for radiusID = 1:length(radiustochoosefrom)
% % %         thisradius = radiustochoosefrom(radiusID);
% % %         num_neighbors(molecule_ID,radiusID) = sum(diff_thisXY<thisradius)-1;
% % %     end
% % % end
% % % toc
% % % 
% % % %% Make scatter plot with color as density
% % % 
% % % figmau
% % % scatter(X,Y,10,num_neighbors(:,3),'filled')
% % % xlabel('X (nm)')
% % % ylabel('Y (nm)')
% % % title('Scatter plot. Color is num neighbors')
% % % axis image xy
% % % fig2pretty
% % % colormap(parula)
% % % colorbar
% % % caxis([0 50])
% % % 
% % % %% Save figure
% % % figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

% Make 2D histogram
binsizenm = 32;
h = hist3([Y X],'Ctrs',{min(Y):binsizenm:max(Y) min(X):binsizenm:max(X)});
imagescmau(h)
clear h
axis image xy
title('Histogram of all weighted centroids')
fig2pretty
% colormap(inferno)
colormap(flipud(inferno))
caxis([0 5])


% Save figure
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);




%% v3 change. REMOVE DUPLICATES BEFORE CLUSTERING
num_molecules = numel(X);

frame_threshold_log = zeros(num_molecules,1);
distance_threshold_log = zeros(num_molecules,1);

max_dist_to_check = 160; % nm
max_frames_to_check = 100; % number of frames

for ii = 1:num_molecules
    
    thisX = X(ii); % X for this molecule
    thisY = Y(ii); % Y for this molecule
    thisXY_diff = sqrt( (X-thisX).^2 + (Y-thisY).^2 ); % distance to other molecules
    thisXY_diff(thisXY_diff==0) = max_dist_to_check * 2; % omit distance to ownself
    closeby_neighbors_IND = find(thisXY_diff<max_dist_to_check);

    if numel(closeby_neighbors_IND) > 0 % if there are molecules close by
        closeby_frames_diff = frame1(closeby_neighbors_IND) - frame1(ii);
        closeby_frames_diff(closeby_frames_diff<=0) = max_frames_to_check*2; % omit current frame and earlier frames
        [frame_threshold_log(ii) ,closestneighbor_IND] = min(closeby_frames_diff); % difference in frames
        distance_threshold_log(ii) = abs(thisXY_diff(closeby_neighbors_IND(closestneighbor_IND)));

        
    end
end

% Make a colorbar for the plot below
density_color_log = zeros(size(frame_threshold_log));
parfor ii = 1:numel(density_color_log)
    thisframe = frame_threshold_log(ii);
    thisdistance = distance_threshold_log(ii);
%     molecules_with_this_frame_IND = find(frame_threshold_log == thisframe);
%     distances_to_check = distance_threshold_log(molecules_with_this_frame_IND);

    distances_to_check = distance_threshold_log(frame_threshold_log == thisframe);
    
    density_color_log(ii) = numel(find(abs(distances_to_check-thisdistance) < 10));
end

%% Scatter number of frames and distance to "nearest neighbor"
figmau
% scatter(frame_threshold_log , distance_threshold_log,'w.')
scatter(frame_threshold_log , distance_threshold_log,5,log(density_color_log),'filled')
ylabel('Distance to nearby molecule (nm)')
xlabel('Frames to nearby molecule (frames)')
xlim([0 max_frames_to_check])
% xlim([0 50])
fig2pretty
% set(gca,'Color','k')
colorbar

% scatter distance to neighbor
figmau
ax1 = subplot(1,2,1);
hold on
scatter(X,Y,50,distance_threshold_log,'filled','MarkerEdgeColor','k')
scatter(X(distance_threshold_log==0),Y(distance_threshold_log==0),50,'k','filled','MarkerEdgeColor','k')
xlabel('Color is distance (nm)')
hold off
axis image xy
colorbar

% scatter frame to neighbor
ax2 = subplot(1,2,2);
hold on
scatter(X,Y,50,frame_threshold_log,'filled','MarkerEdgeColor','k')
scatter(X(frame_threshold_log>max_frames_to_check),Y(frame_threshold_log>max_frames_to_check),50,'k','filled','MarkerEdgeColor','k')
xlabel('Color is frames')
hold off
axis image xy
colorbar

linkaxes([ax1,ax2],'xy')

% Save figure
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

% Look at molecules above thresholds
frame_threshold_duplicate = 40;
distance_threshold_duplicate = 120;

duplicate_log = zeros(num_molecules,1);

for ii = 1:numel(X)
    if distance_threshold_log(ii) < distance_threshold_duplicate && ...
            frame_threshold_log(ii) < frame_threshold_duplicate && ...
            distance_threshold_log(ii) > 0 && ...
            frame_threshold_log(ii) > 0
        duplicate_log(ii) = 1;
    end
end

%
figmau
hold on

scatter(X(duplicate_log==1),Y(duplicate_log==1),50,'k','filled')
scatter(X(duplicate_log==0),Y(duplicate_log==0),50,'m','filled')

% scatter(X(duplicate_log==1),Y(duplicate_log==1),'k.')
hold off
axis image xy
fig2pretty
xlabel('K is duplicates. M is non-duplicate')

%
binsizenm = 32;
X_ok = X(duplicate_log==0);
Y_ok = Y(duplicate_log==0);
frame1_ok = frame1(duplicate_log==0);
h = hist3([Y_ok X_ok],'Ctrs',{min(Y_ok):binsizenm:max(Y_ok) min(X_ok):binsizenm:max(X_ok)});
imagescmau(h)
clear h
axis image xy
title('Histogram of all weighted centroids that are not duplicates')
fig2pretty
% colormap(inferno)
colormap(flipud(inferno))
caxis([0 5])

%
disp(['Start with ',num2str(length(X)),' molecules'])

disp(['Num molecules with duplicates = ',num2str(sum(duplicate_log==1)),' molecules'])

disp(['Num molecules with NO duplicates = ',num2str(sum(duplicate_log==0)),' molecules'])

% Save figure
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);




%% MAKE VORONOI PLOT
[ClusterNumberLog] = VoronoiClustering(X_ok,Y_ok); % one function

% [ClusterNumberLog] = VoronoiClustering(X,Y); % one function

% Save figure
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);



%% Check how close the molecules are to each other in a cluster in time
numberofclusters = numel((unique(ClusterNumberLog))) - 1;

[~,max_members] = mode(ClusterNumberLog(ClusterNumberLog>0));

nearestframelog = zeros(numberofclusters,max_members);
nearestframe_distlog = zeros(numberofclusters,max_members);

clustersX = zeros(numberofclusters,max_members);
clustersY = zeros(numberofclusters,max_members);

for i = 1:numberofclusters % for each cluster
    if sum(ClusterNumberLog == i) > 0
        thiscluster_X = X_ok(ClusterNumberLog == i); % all the X values of this cluster
        thiscluster_Y = Y_ok(ClusterNumberLog == i); % all the Y values of this cluster
        thiscluster_frame = zeros(numel(thiscluster_X),1); % all the frames for this cluster

        for j = 1:numel(thiscluster_X)
            [~,this] = min((X_ok - thiscluster_X(j)).^2 + ...
                (Y_ok - thiscluster_Y(j)).^2 );
            thiscluster_frame(j) = frame1_ok(this); % all the frames for this cluster
        end
        
        [thiscluster_frame_sorted , sorted_IND] = sort(thiscluster_frame); % sort according to nearest frame
        thiscluster_X_sorted = thiscluster_X(sorted_IND);
        thiscluster_Y_sorted = thiscluster_Y(sorted_IND);
        
        thiscluster_X_sorted_diff = diff(thiscluster_X_sorted);
        thiscluster_Y_sorted_diff = diff(thiscluster_Y_sorted);
        thiscluster_distances = sqrt(thiscluster_X_sorted_diff.^2+...
            thiscluster_Y_sorted_diff.^2); % distance bewteen each molecule in cluster and subsequent molecule in cluster
        
        nearestframelog(i,2:numel(thiscluster_frame_sorted)) = diff(thiscluster_frame_sorted);
        nearestframe_distlog(i,2:numel(thiscluster_distances)+1) = thiscluster_distances;
        clustersX(i,1:numel(thiscluster_X)) = thiscluster_X;
        clustersY(i,1:numel(thiscluster_Y)) = thiscluster_Y;
    end
end


%% Look at time between molecules in each cluster
nearestframelog_numbers = nearestframelog(nearestframelog>0);
nearestframe_distlog_numbers = nearestframe_distlog(nearestframelog>0);

figmau
hist(nearestframelog_numbers(nearestframelog_numbers<200),1:200)
xlabel('#frames to next spot in same cluster')
fig2pretty

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

% scatter distance to next spot in same cluster vs num frames to that spot
figmau
scatter(nearestframe_distlog_numbers,nearestframelog_numbers,'k.')
xlabel('Dist to next spot in same cluster (nm)')
ylabel('#frames to next spot in same cluster')
ylim([0 100])
xlim([0 200])
fig2pretty

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);








%% Find number of members of each cluster
numbermembersincluster_log = zeros(numberofclusters,1);
% uniqueclusternumberlog = unique(clusternumber_log);
for i = 1:numberofclusters
    %     thisclusternumber = uniqueclusternumberlog(i+1);
    thisclusternumber = i;
    numbermembersincluster_log(i) = numel(find(ClusterNumberLog==thisclusternumber));
end



% scatter
colormap787 = parula(ceil(max(numbermembersincluster_log)));
figmau
hold on
for i = 1:numberofclusters
    if numbermembersincluster_log(i)>4
                scatter(X_ok(ClusterNumberLog==i),...
                    Y_ok(ClusterNumberLog==i), ...
                    50,colormap787(round(numbermembersincluster_log(i)),:),'filled','MarkerEdgeColor','k');
%         scatter(X_ok(ClusterNumberLog==i),...
%             Y_ok(ClusterNumberLog==i), ...
%             50,'b','filled','MarkerEdgeColor','k');
    end
end
hold off
xlabel('X (nm)')
ylabel('Y (nm)')
axis image ij
colormap(parula)
colorbar
fig2pretty

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);


%%
figmau
hist(numbermembersincluster_log(numbermembersincluster_log>4),0:1:max(numbermembersincluster_log(numbermembersincluster_log>4)))
xlabel('#molecules in each cluster')
ylabel('#clusters within each bin')
xlim([0 max(numbermembersincluster_log)])
fig2pretty

% a number of members threshold at 4 seems to be good


%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

% Find area of each cluster
areaincluster_log = zeros(numberofclusters,1);
for i = 1:numberofclusters
    
    thiscluster_X = X_ok(ClusterNumberLog == i);
    thiscluster_Y = Y_ok(ClusterNumberLog == i);
    if numel(thiscluster_X) > 4
        [~,thisarea] = convhull(double(thiscluster_X),double(thiscluster_Y));
        areaincluster_log(i) = thisarea;
        %         figmau
        %         scatter(thiscluster_X-min(thiscluster_X),thiscluster_Y-min(thiscluster_Y),'k.')
        %         axis image  xy
    end
end

%
figmau
hist(areaincluster_log(areaincluster_log>0),0:10:max(areaincluster_log))
xlabel('Area of each cluster (nm^2)')
ylabel('#clusters within each bin')
xlim([0 max(areaincluster_log)])
fig2pretty

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%
% Scatter plot of number of molecules vs area
figmau
scatter(numbermembersincluster_log(numbermembersincluster_log>4), ...
    areaincluster_log(numbermembersincluster_log>4),15,'kx')


[X791, X791ind] = sort(numbermembersincluster_log(numbermembersincluster_log>4));
Y791 = areaincluster_log(numbermembersincluster_log>4);
Y791 = Y791(X791ind);

% % % fun = @(r) r(1) + r(2)*(X791.^(2/3)) - Y791;
% % % r0 = [ 0.5 0.5 ];
% % % r99 = lsqnonlin(fun,r0,[],[],options);
% % % hold on
% % % plot(X791,r99(1) + r99(2)*X791.^(2/3),'r')
% % % scatter(numbermembersincluster_log(numbermembersincluster_log>4), ...
% % %     areaincluster_log(numbermembersincluster_log>4),15,'kx')
% % % hold off

ft = fittype({'x.^(2/3)','1'});
myfit = fit(X791,Y791,ft);
hold on
plot(myfit,'r')
scatter(numbermembersincluster_log(numbermembersincluster_log>4), ...
    areaincluster_log(numbermembersincluster_log>4),20,'ko','filled')
hold off

xlabel('#molecules in cluster')
ylabel('Area in cluster (nm^2)')
fig2pretty
ylim([0 max(areaincluster_log)+100])

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);



%% Radial distribution for SUPER-RESOLUTION
% Look at the scatter plot of the XY coordinates after the duplicates have
% been removed
figure
scatter(X_ok,Y_ok,'k.')
axis image xy

% import DNA channel
[dataFile1, dataPath] = uigetfile({'*.tif';'*.*'},'Open file for main file');
if isequal(dataFile1,0), error('User cancelled the program'); end
% Get some info about the dark counts flie
dataFile = [dataPath dataFile1];
disp(['File name = ', (dataFile)])
dataFileInfo = imfinfo(dataFile,'tif');
total_num_frames = length(dataFileInfo);
imgHeight = dataFileInfo.Height;
imgWidth = dataFileInfo.Width;

im = tiffread22(dataFile,1,total_num_frames); % tiffread22 is actually tiffread with some things commented out
DATA = zeros(imgHeight,imgWidth,total_num_frames);
for i = 1:total_num_frames
        DATA(:,:,i) = double(im(i).data);
end

% These frames are usually cropped and concatenated in imageJ before import
DATA = single(DATA); % single uses less memory than double

% Look at green DNA image
imagescmau(DATA(:,:,1))

%% Crop only nucleus.
DATA_crop = sum(DATA(:,:,1:50),3);
DATA_crop = DATA_crop./max(DATA_crop(:));
DATA_crop = imcrop(DATA_crop./max(DATA_crop(:)));
% DATA_crop = flipud(DATA_crop);

%% Look at both images
figmau
subplot(1,2,1)
imagesc(DATA_crop)
axis image xy
subplot(1,2,2)
scatter(X_ok,Y_ok,'k.')
axis image xy

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% Make BW mask
DATA_crop_BW = DATA_crop;
figure
hist(DATA_crop_BW(:),500)
% ylim([0 5E3])
[thresholdforBWx , ~ ] = ginput(1);

DATA_crop_BW = DATA_crop_BW>thresholdforBWx;
DATA_crop_BW = bwareaopen(DATA_crop_BW,5);
DATA_crop_BW = imfill(DATA_crop_BW,4,'holes');
DATA_crop_BW = imclearborder(DATA_crop_BW);

imagescmau(DATA_crop_BW)

%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);


%% Check if XY divided by pixelsizenm is in BW
X_ok_in_nucleus = X_ok;
Y_ok_in_nucleus = Y_ok;
pixelsizenm = 160;

X_translate_pixel = 12;
Y_translate_pixel = 1;

BW_X_pixelcheck = X_ok_in_nucleus/pixelsizenm + X_translate_pixel;
BW_Y_pixelcheck = Y_ok_in_nucleus/pixelsizenm + Y_translate_pixel;
    

figmau
hold on
imagesc(DATA_crop_BW)
axis image xy
scatter(BW_X_pixelcheck,BW_Y_pixelcheck,10,'k','filled')
axis image xy
hold off

% if outside nucleus, set BW_X_pixelcheck to 0
for ii = 1:numel(X_ok)
    if round(BW_Y_pixelcheck(ii)) < size(DATA_crop_BW,1) && ...
            round(BW_X_pixelcheck(ii)) < size(DATA_crop_BW,2)
    if DATA_crop_BW(round(BW_Y_pixelcheck(ii)),round(BW_X_pixelcheck(ii))) == 0
        BW_X_pixelcheck(ii) = 0;
        BW_Y_pixelcheck(ii) = 0;
    end
    end
end


figmau
hold on
imagesc(DATA_crop_BW)
axis image xy
scatter(BW_X_pixelcheck,BW_Y_pixelcheck,10,'k','filled')
axis image xy
hold off

%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);


%% Do radial distribution
% Find middle of nucleus
stats = regionprops(DATA_crop_BW,'centroid');
middle_of_nucleus = stats.Centroid;
figure
imagesc(DATA_crop_BW)
hold on
plot(middle_of_nucleus(1),middle_of_nucleus(2),'kx')
hold off
axis image xy

%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% For each molecule, find distance to middle, find distance to edge
distance_to_middle = zeros(size(X_ok));
distance_to_edge = zeros(size(X_ok));

zero_in_BW = find(DATA_crop_BW==0);
[zero_in_BW_Y,zero_in_BW_X] = ind2sub(size(DATA_crop_BW),zero_in_BW);

for ii = 1:numel(BW_X_pixelcheck)
    thisX = BW_X_pixelcheck(ii); % pixel space
    thisY = BW_Y_pixelcheck(ii); % pixel space
    distance_to_middle(ii) = sqrt( (middle_of_nucleus(1)-thisX)^2 + (middle_of_nucleus(2)-thisY)^2 ).*pixelsizenm; % nm space
    
    %% find closest 0 in BW image
    distances_to_zero = sqrt( (zero_in_BW_X-thisX).^2 + (zero_in_BW_Y-thisY).^2 ).*pixelsizenm; % nm space
    [distance_to_edge(ii),~] = min(distances_to_zero);
    
end

%% Look at scatter with color as distance from middle or edge
figmau
scatter(BW_X_pixelcheck(BW_X_pixelcheck~=0),BW_Y_pixelcheck(BW_X_pixelcheck~=0),10, ...
    distance_to_middle(BW_X_pixelcheck~=0),'filled')
title('Distance to middle (nm)')
fig2pretty
grid off
colorbar
axis image xy

figmau
scatter(BW_X_pixelcheck(BW_X_pixelcheck~=0),BW_Y_pixelcheck(BW_X_pixelcheck~=0),10,...
    distance_to_edge(BW_X_pixelcheck~=0),'filled')
title('Distance to edge (nm)')
fig2pretty
grid off
colorbar
axis image xy

figmau
scatter(BW_X_pixelcheck(BW_X_pixelcheck~=0),BW_Y_pixelcheck(BW_X_pixelcheck~=0),10,...
    distance_to_edge(BW_X_pixelcheck~=0)./ ...
    (distance_to_edge(BW_X_pixelcheck~=0)+distance_to_middle(BW_X_pixelcheck~=0)),'filled')
title('Normalized distance to edge (nm)')
fig2pretty
grid off
colorbar
axis image xy
%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% Make graphs with distance from edge
figmau
hist(distance_to_edge(BW_X_pixelcheck~=0),500)
title('Histogram of distance to edge (nm)')
xlabel('Distance to edge (nm)')
fig2pretty
grid off

% make graphs with normalized distance from edge
figmau
hist(distance_to_edge(BW_X_pixelcheck~=0)./ ...
    (distance_to_edge(BW_X_pixelcheck~=0)+distance_to_middle(BW_X_pixelcheck~=0)),500)
title('Normalized Histogram of distance to edge (nm)')
xlabel('Distance to edge (nm)')
fig2pretty
grid off


%%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);




%% Do clustering only on those in nucleus

% MAKE VORONOI PLOT
X_nucleus_nm = BW_X_pixelcheck(BW_X_pixelcheck~=0)*pixelsizenm;
Y_nucleus_nm = BW_Y_pixelcheck(BW_X_pixelcheck~=0)*pixelsizenm;
[ClusterNumberLog_nucleus] = VoronoiClustering(X_nucleus_nm, ...
     Y_nucleus_nm); % one function

% Save figure
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);


%% Find number of members of each cluster
numberofclusters_nucleus = numel((unique(ClusterNumberLog_nucleus))) - 1;
numbermembersincluster_log_nucleus = zeros(numberofclusters_nucleus,1);
% uniqueclusternumberlog = unique(clusternumber_log);
for i = 1:numberofclusters_nucleus
    %     thisclusternumber = uniqueclusternumberlog(i+1);
    thisclusternumber = i;
    numbermembersincluster_log_nucleus(i) = numel(find(ClusterNumberLog_nucleus==thisclusternumber));
end


% scatter
colormap787 = parula(ceil(max(numbermembersincluster_log_nucleus)));
figmau
hold on
for i = 1:numberofclusters_nucleus
    if numbermembersincluster_log_nucleus(i)>4
                scatter(X_nucleus_nm(ClusterNumberLog_nucleus==i),...
                    Y_nucleus_nm(ClusterNumberLog_nucleus==i), ...
                    50,colormap787(round(numbermembersincluster_log_nucleus(i)),:),'filled','MarkerEdgeColor','k');
%         scatter(X_ok(ClusterNumberLog==i),...
%             Y_ok(ClusterNumberLog==i), ...
%             50,'b','filled','MarkerEdgeColor','k');
    end
end
hold off
xlabel('X (nm)')
ylabel('Y (nm)')
axis image ij
colormap(parula)
colorbar
fig2pretty

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);


%%
figmau
hist(numbermembersincluster_log_nucleus(numbermembersincluster_log_nucleus>4),0:1:max(numbermembersincluster_log_nucleus(numbermembersincluster_log_nucleus>4)))
xlabel('#molecules in each cluster')
ylabel('#clusters within each bin')
xlim([0 max(numbermembersincluster_log_nucleus)])
fig2pretty

% a number of members threshold at 4 seems to be good


%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);

%% Find area of each cluster
areaincluster_log_nucleus = zeros(numberofclusters_nucleus,1);
for i = 1:numberofclusters_nucleus
    
    thiscluster_X = X_nucleus_nm(ClusterNumberLog_nucleus == i);
    thiscluster_Y = Y_nucleus_nm(ClusterNumberLog_nucleus == i);
    if numel(thiscluster_X) > 4
        [~,thisarea] = convhull(double(thiscluster_X),double(thiscluster_Y));
        areaincluster_log_nucleus(i) = thisarea;
        %         figmau
        %         scatter(thiscluster_X-min(thiscluster_X),thiscluster_Y-min(thiscluster_Y),'k.')
        %         axis image  xy
    end
end

%
figmau
hist(areaincluster_log_nucleus(areaincluster_log_nucleus>0),0:10:max(areaincluster_log_nucleus))
xlabel('Area of each cluster (nm^2)')
ylabel('#clusters within each bin')
xlim([0 max(areaincluster_log_nucleus)])
fig2pretty

%
figurecounter = savefigure(figurecounter,pathtoanalysisfolder);






%%

%%
