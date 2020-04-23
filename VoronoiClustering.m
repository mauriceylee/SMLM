function [ClusterNumberLog] = VoronoiClustering(X,Y)

%% Look at XY in patches first
%% INITIAL CLUSTERING of experimental data with VoronoiN
[Vertex_allcells,Vert_eachcell] = voronoin(double([X Y]));

% V is a numv-by-n array of the numv Voronoi vertices in n-dimensional space,
% each row corresponds to a Voronoi vertex. C is a vector cell array where
% each element contains the indices into V of the vertices of the corresponding Voronoi cell.

%  Calculate volume in each Voronoi cell
Vol_expt = zeros(size(X));
for jj = 1:length(Vert_eachcell)
    if all(Vert_eachcell{jj}~=1)
        VertCell = Vertex_allcells(Vert_eachcell{jj},:);
    else
        VertCell = Vertex_allcells(Vert_eachcell{jj}(2:end),:);
    end
    
    try
        [~,Vol_expt(jj,1)] = convhulln(VertCell,{'Qt','Pp'});
    catch
        Vol_expt(jj,1)=nan;
    end
end

%% Initial area/volume calculation
vol_lim = 1000; % this is just an upper limit for the histogram later
numbins = 500;
figmau
hist(Vol_expt(Vol_expt<vol_lim),numbins)
xlim([0 vol_lim])
xlabel('Area of each voronoi cell (nm^2)')
fig2pretty
[Vol_threshold,~] = ginput(1)
hold on
plot([Vol_threshold Vol_threshold],[0 500],'r')
hold off

%% Histogram of volume or area but in -log
numbins = 500;
figmau
hist(-log(Vol_expt),numbins)
% xlim([0 vol_lim])
xlabel('Area of each voronoi cell (nm^2)')
fig2pretty



%% VISUALIZE the voronoi plot with color as density but using PATCHES instead
figmau
for i = 1:length(Vert_eachcell)
    if all(Vert_eachcell{i}~=1)   % If at least one of the indices is 1,
        % then it is an open region and we can't patch that.
%         patch( Vertex_allcells(Vert_eachcell{i},1) , Vertex_allcells(Vert_eachcell{i} , 2), ...
%             -log(Vol_expt(i)),'EdgeColor','none'); % use color i.
        patch( Vertex_allcells(Vert_eachcell{i},1) , Vertex_allcells(Vert_eachcell{i} , 2), ...
            -log(Vol_expt(i)),'EdgeColor','none'); % use color i.
    end
end
colormap(parula)
% colormap(dusk)
colorbar
axis image xy
xlim([min(X) max(X)])
ylim([min(Y) max(Y)])
xlabel('X (nm)')
ylabel('Y (nm)')
caxis([-12 -4])
fig2pretty


%% Set threshold of 100 nm^2 for density of Voronoi cells that are considered clustered

% any voronoi patch smaller than this threshold is considered clustered

% choose low area cells, check which have vertices touching
max_vertex = 12;
vertexforeachcell = zeros(numel(Vol_expt),max_vertex,'single');
for i = 1:numel(Vol_expt)
    thisvertices = Vert_eachcell{(i)};
    vertexforeachcell(i,1:numel(thisvertices)) = thisvertices;
end

vertexforeachcell(vertexforeachcell==0) = NaN;


%% Assign cluster numbers to the Voronoi cells that are considered small

% initialize a log to contain the ID of each cluster
ClusterNumberLog = zeros(numel(Vol_expt),1);

% give a cluster number to all small cells
small_molecule_ID = find(Vol_expt<Vol_threshold);
ClusterNumberLog(small_molecule_ID) = small_molecule_ID; % row numbers of patches smaller than threshold


%% Prepare for clustering small voronoi patches that are touching
ClusterNumberLog_small_IND = find(ClusterNumberLog>0); % find small patches
ClusterNumberLog_small = ClusterNumberLog(ClusterNumberLog_small_IND); % use only the small patches
ClusterNumberLog_small = repmat(ClusterNumberLog_small,[1 size(vertexforeachcell,2)]); % convert into same size as vertexforeachcell_small

vertexforeachcell_small = vertexforeachcell(ClusterNumberLog_small_IND,:); % only vertices from small patches

% reshape for parfor loop
vertexforeachcell_small_reshape = reshape(vertexforeachcell_small,[numel(vertexforeachcell_small),1]);
ClusterNumberLog_small_reshape = reshape(ClusterNumberLog_small,[numel(vertexforeachcell_small),1]);

ClusterNumberLog_small_reshape_new = ClusterNumberLog_small_reshape;

disp(['Start with ',num2str(numel(unique(ClusterNumberLog_small_reshape))),' clusters.'])

%%
changeiszero = 1;

while changeiszero > 0
    %% Change cluster numbers to be the smallest number for touching patches
    parfor ii = 1 :size(vertexforeachcell_small_reshape,1)
        if ~isnan(vertexforeachcell_small_reshape(ii))
            ClusterNumberLog_small_reshape_new(ii) = min(ClusterNumberLog_small_reshape(vertexforeachcell_small_reshape == ...
                vertexforeachcell_small_reshape(ii)));
        end
    end
    %%
    disp(['Now with ',num2str(numel(unique(ClusterNumberLog_small_reshape_new))),' clusters.'])
    
    %% Check if the new cluster numbers are same as before
    ClusterNumberLog_small_reshape_new = reshape(ClusterNumberLog_small_reshape_new , size(vertexforeachcell_small));
    ClusterNumberLog_small_reshape_new = min(ClusterNumberLog_small_reshape_new,[],2);
    ClusterNumberLog_small_reshape_new = repmat(ClusterNumberLog_small_reshape_new,[size(vertexforeachcell,2) 1]);

    changeiszero = max((ClusterNumberLog_small_reshape_new - ClusterNumberLog_small_reshape).^2);

    ClusterNumberLog_small_reshape = ClusterNumberLog_small_reshape_new;
    
end



%% Scatter only small molecules with color as cluster number
figmau
hold on
scatter(X,Y,50,'k','filled')
scatter(X(ClusterNumberLog_small_IND),Y(ClusterNumberLog_small_IND), ...
    50,ClusterNumberLog_small_reshape_new(1:numel(ClusterNumberLog_small_IND)),'filled')
hold off
axis image xy
colormap(lines)


%% Add nearby molecules

% expand ClusterNumberLog_small_reshape to include patches that were not small enough
ClusterNumberLog(small_molecule_ID) = ClusterNumberLog_small_reshape_new(1:numel(ClusterNumberLog_small_IND));

% Change the cluster numbers to be the unique values without space between
% them
uniqueclusternumbers = unique(ClusterNumberLog);
for ii = 2:numel(uniqueclusternumbers)
    ClusterNumberLog(ClusterNumberLog==uniqueclusternumbers(ii)) = ii-1;
end
%%
changeiszero = 1;
while changeiszero > 0
    
    oldClusterNumberLog = ClusterNumberLog;
    
    for ii = 160 % 1:max(ClusterNumberLog) % for each cluster
        
        % for each cluster, find the max nearest neighbor
        thisclusterX = X(ClusterNumberLog==ii);
        thisclusterY = Y(ClusterNumberLog==ii);
        maxNNdist = 0;
        for jj = 1:numel(thisclusterX)
            diffX_1 = thisclusterX - thisclusterX(jj);
            diffY_1 = thisclusterY - thisclusterY(jj);
            diffXY_1 = sqrt(diffX_1.^2+diffY_1.^2);
            diffXY_NN = min(diffXY_1(diffXY_1>0));
            maxNNdist = max([maxNNdist diffXY_NN]);
        end
        
        % check X and Y and change their cluster number if need be
        
        for jj = 1:numel(thisclusterX)
            diffX_2 = X - thisclusterX(jj);
            diffY_2 = Y - thisclusterY(jj);
            diffXY_2 = sqrt(diffX_2.^2+diffY_2.^2);
            ClusterNumberLog(diffXY_2<maxNNdist) = ii;
        end
    end
    
    newClusterNumberLog = ClusterNumberLog;
    changeiszero = sum((newClusterNumberLog-oldClusterNumberLog).^2);
    disp(['Current change = ',num2str(changeiszero)])
end

%% Scatter only small molecules with color as cluster number
figmau
hold on
scatter(X(ClusterNumberLog>0),Y(ClusterNumberLog>0),50,ClusterNumberLog(ClusterNumberLog>0),'filled')
scatter(X(ClusterNumberLog==0),Y(ClusterNumberLog==0),50,'k','filled')
hold off
axis image xy
colormap(lines)

%% find cluster number. troubleshooting
% % % Xdiff_now = X-3787;
% % % Ydiff_now = Y-8407;
% % % [~,molecule_ID_now] = min(Ydiff_now.^2 + Xdiff_now.^2);
% % % ClusterNumberLog(molecule_ID_now)

















































%%