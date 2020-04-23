function imagescmau(varargin)
% Get screensize and get figure
figure('units','normalized','outerposition',[0 0 1 1],'color','w');

load inferno.mat
%% Imagesc images 
if size(varargin,2)==1
    imagesc(varargin{1})
    colormap(inferno)
    colorbar
    axis image xy
elseif size(varargin,2)==2
    for i = 1:size(varargin,2)
        subplot(1,size(varargin,2),i)
        imagesc(varargin{i})
        colormap(inferno)
        title(['Image ',num2str(i)])
        colorbar
        axis image xy
    end
elseif size(varargin,2)==3
    for i = 1:size(varargin,2)
        subplot(2,2,i)
        imagesc(varargin{i})
        colormap(inferno)
        title(['Image ',num2str(i)])
        colorbar
        axis image xy
    end
elseif size(varargin,2)==4
    for i = 1:size(varargin,2)
        subplot(2,2,i)
        imagesc(varargin{i})
        colormap(inferno)
        title(['Image ',num2str(i)])
        colorbar
        axis image xy
    end
elseif size(varargin,2)==5
    for i = 1:size(varargin,2)
        subplot(2,3,i)
        imagesc(varargin{i})
        colormap(inferno)
        title(['Image ',num2str(i)])
        colorbar
        axis image xy
    end
elseif size(varargin,2)==6
    for i = 1:size(varargin,2)
        subplot(2,3,i)
        imagesc(varargin{i})
        colormap(inferno)
        title(['Image ',num2str(i)])
        colorbar
        axis image xy
    end
end
fig2pretty

