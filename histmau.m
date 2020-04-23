function histmau(values)
values = double(values);
figure('units','normalized','outerposition',[0 0 1 1],'color','w');
num = numel(values);
maxbars = 200;
if sqrt(num)<maxbars
    hist(values,sqrt(num))
else
    hist(values,maxbars)
end
title('Histogram of values')
ylabel('Number of occurences')
fig2pretty
end