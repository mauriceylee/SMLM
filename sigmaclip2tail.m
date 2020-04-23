function [highthresholdtobeused,lowthresholdtobeused,logicalforwithinthreshold] = sigmaclip2tail(givendistribution,num_sigma,num_iterations)
%% Introduction
% This code calculates an appropriate threshold to be used in any
% distribution by iteratively calculating the median and sigma of the
% distribution SMALLER than the threshold

% Example usage
% [Iwanttousethisthreshold, thesearetheLOWERvalues] = sigmaclip(weirdlookingdistribution, 3, 10);

%% Default input if only 1 or 2 arguments are given
default_num_sigma = 3;
default_num_iterations = 20;
if nargin == 1
    num_sigma = default_num_sigma;
    num_iterations = default_num_iterations;
elseif nargin == 2
    num_iterations = default_num_iterations;
elseif nargin == 3
end

%% Loop to find the appropriate threshold
logicalforwithinthreshold = true(size(givendistribution));
% 1 is lower than threshold, 0 is higher than threshold
% threshold can only decrease with each iteration
for ii = 1:num_iterations
    median_distribution = median(givendistribution(logicalforwithinthreshold));
    sigma_distribution = std(givendistribution(logicalforwithinthreshold));
    
    % calculate the appropriate threshold for this iteration
    highthresholdtobeused = median_distribution+num_sigma*sigma_distribution;
    lowthresholdtobeused = median_distribution-num_sigma*sigma_distribution;
    % change the logical for higher values to zero
    logicalforwithinthreshold = true(size(givendistribution));
    logicalforwithinthreshold(givendistribution>highthresholdtobeused) = 0;
    logicalforwithinthreshold(givendistribution<lowthresholdtobeused) = 0;
    
end


%% Troubleshoot or check out the histogram
% % % figure(38)
% % % numbins = 500;
% % % h38 = hist(givendistribution,numbins);
% % % plot(linspace(min(givendistribution),max(givendistribution),numbins),h38)
% % % hold on
% % % plot([highthresholdtobeused highthresholdtobeused],[0 max(h38)],'r')
% % % plot([lowthresholdtobeused lowthresholdtobeused],[0 max(h38)],'r')
% % % hold off
% % % xlabel('Values of given distribution')
% % % title('Histogram of distribution. Red vertical line is threshold')

end
