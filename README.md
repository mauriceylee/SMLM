# SMLM
Maurice's SMLM matlab code

This is just some SMLM matlab code that may or may not be useful.
NOTE: IT IS VERY MESSY!
Contact me on Twitter @maurice_y_lee for more info or if you have any questions.

The main code for converting single-molecule blinking movies to a table of parameters (much like ThunderSTORM) is in sPSF_SM_blinking_v34_forGithub.m. Do note that this code does not have multi-emitter fitting.

--- HOW TO USE sPSF_SM_blinking_v34_forGithub.m ---
1. Place all the scripts into the same folder
2. The tif image files have to be cropped and concatenanted before using this MATLAB script
3. Remember to have the dark counts saved too. This is a tif image file that is about 500 frames that the camera takes with the shutter closed with the exact exposure and gain settings with the same cropped area.
4. When you click run at the top of the MATLAB window, the script will run.
5. There will be a box that shows up to choose several parameters to tweak.
6. The code will prompt you to import the dark counts
7. The code will prmopt you to import the main tif files with the single-molecule blinking data

---------------------------------------------------

tiffread2 or tiffread22 are not written by me!
They are written by Francois Nedelec who is now at Cambridge University (francois.nedelec {a t} slcu {d ot} cam.ac.uk).

I think I made some changes to tiffread2 or tiffread22 several years ago. But I don't remember.
They import tiff stacks into MATLAB very quickly.

fig2pretty is a script that makes the figures look better. This script is also not written by me! I think my friend Dr Alex Colavin wrote it many years ago.

singleIntegratedGaussianOffset_symmetricversion.m is based on code that Dr Lucien Weiss wrote or that he passed to me.

The Voronoi clustering code is based on code that Dr Camille Bayas wrote.

analyzeXYFrame_v3.m tries to do clustering by combining Voronoi clustering and a modified version of DBScan based on the Voronoi clustering. Basically, it uses the Voronoi clustering data to choose the parameters to add localizations that are near to clusters to be included within those clusters. This helps to solve the problematic situation where there are occasional localizations that have Voronoi cell areas that are too large to be considered clustered, but are actually near enough to a cluster to be considered a part of that cluster. This modified clustering algorithm is not published. If someone who reads this thinks the algorithm is publishable, by all means, please go ahead.
