# SMLM
Maurice's SMLM matlab code

This is just some SMLM matlab code that may or may not be useful.
NOTE: IT IS VERY MESSY!
Contact me on Twitter @maurice_y_lee for more info or if you have any questions.

The main code for converting single-molecule blinking movies to a table of parameters (much like ThunderSTORM) is in sPSF_SM_blinking_v34_forGithub.m

tiffread2 or tiffread22 are not written by me!
They are written by Francois Nedelec % nedelec (at) embl.de
% Cell Biology and Biophysics, EMBL; Meyerhofstrasse 1; 69117 Heidelberg; Germany
% http://www.embl.org
% http://www.cytosim.org

I think I made some changes to tiffread2 or tiffread22 several years ago. But I don't remember.
They import tiff stacks into MATLAB very quickly.

fig2pretty is also not written by me! I think my friend Dr Alex Colavin wrote it many years ago.

singleIntegratedGaussianOffset_symmetricversion.m is based on code that Dr Lucien Weiss wrote or that he passed to me.

analyzeXYFrame_v3.m tries to do clustering by combining Voroinoi clustering and a modified version of DBScan based on the Voronoi clustering. Basically, it uses the Voronoi clustering data to choose the parameters to add localizations that are near to clusters to be included within those clusters. This helps to solve the problem where there are occasional localizations that have Voronoi cell areas that are too large to be considered clustered, but are actually near enough to a cluster to be considered cluster.
This modified clustering algorithm is not published. If someone who reads this thinks the algorithm is publishable, by all means, please go ahead.
