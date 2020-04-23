function err = singleIntegratedGaussianOffset_symmetricversion(par,Zdata,ii,jj)

%   par(1) = amplitude
%   par(2) = Xo
%   par(3) = Yo
%   par(4) = sigmaX;
%   par(5) = sigmaY;, NOT USED
%   par(6) = theta; degrees, NOT USED
%   par(7) = DC;

% par = fitparam;
% % par(2) = 5.5; par(3) = 5.5;
% Zdata = double(ROItofitnow);
% hold on, plot(par), plot(upperBound), plot(lowerBound), hold off

%% Calculate the distribution of the pixelated Gaussian PSF (no theta)
% Reference : https://media.nature.com/original/nature-assets/nmeth/journal/v7/n5/extref/nmeth.1449-S1.pdf
% no par(5) which is sigmaY

E_x = 0.5 * ( erf( (ii-par(2)+0.5) / (2*par(4)*par(4)) ) - erf( (ii-par(2)-0.5) / (2*par(4)*par(4)) ));
E_y = 0.5 * ( erf( (jj-par(3)+0.5) / (2*par(4)*par(4)) ) - erf( (jj-par(3)-0.5) / (2*par(4)*par(4)) ));
mu = par(1).*E_x.*E_y + par(7);

err = reshape(mu-Zdata,1,[]);

%% Calculate the distribution of the pixelated Gaussian PSF (with theta)
% Reference for using theta: https://www.mathworks.com/matlabcentral/fileexchange/31485-auto-gaussian---gabor-fits
% % % % % theta = par(6)*pi/180; % theta here is in radians
% % % % % 
% % % % % E_x = 0.5*( erf( ((ii-par(2))*cos(theta) + (jj-par(3))*sin(theta) + 0.5) / (2*par(4)*par(4))) - ...
% % % % %     erf( ((ii-par(2))*cos(theta) + (jj-par(3))*sin(theta) - 0.5) / (2*par(4)*par(4))));
% % % % % E_y = 0.5 * ( erf( (-(ii-par(2))*sin(theta) + (jj-par(3))*cos(theta) + 0.5) / (2*par(5)*par(5))) - ...
% % % % %     erf( (-(ii-par(2))*sin(theta) + (jj-par(3))*cos(theta) - 0.5) / (2*par(5)*par(5))));
% % % % % 
% % % % % mu = par(1).*E_x.*E_y + par(7);
% % % % % 
% % % % % % pause,  imagescmau(Zdata,mu,Zdata-mu)
% % % % % 
% % % % % err = reshape(mu-Zdata,1,[]);
