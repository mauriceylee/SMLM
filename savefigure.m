function figurecounter=savefigure(figurecounter,pathtoanalysisfolder)
h =  findobj('type','figure');
numfigs = length(h);
figSize = [21, 29];            % [width, height]
figUnits = 'Centimeters';
for f = 1:numfigs
    if ishandle(f)==1
        fig = figure(f);
        % Resize the figure
        set(fig, 'Units', figUnits);
        pos = get(fig, 'Position');
%         pos = [pos(1), pos(4)+figSize(2), pos(3)+figSize(1), pos(4)];
        set(fig, 'Position', pos);
        % Output the figure
        filename = sprintf(['Figure',num2str(figurecounter),'_',num2str(f),'.png']);
        %         print( fig , '-dpng', filename );
        %         movefile(filename,pathtoanalysisfolder);
        saveas(fig,[pathtoanalysisfolder,'\', filename])
        close(f)
    end
end
figurecounter = figurecounter+1;

end