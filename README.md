# TGDA
Tensor Grassmann Discriminant Analysis (TGDA)
Weizman action data was provided through mat files for 10 actions included bending (bend.mat), jacking (jack.mat), jumping (jump.mat), jumping in places (pjump.mat), running (run.mat), gallopingside ways (side.mat), skipping (skip.mat), walking (walk.mat), single-hand waving (wave1.mat), and both hands waving (wave2.mat).
The data set was pre-processed by using a people detector. We also grayscaled and resized each video to 20 × 20 × 20. 
All tensor operators ( t-SVD, t-eig, etc.) were provided in wavelet.py file.
Download all files and run weizman_grassmann_kernel.py file to run tensor Grassmann discriminant analysis (TDGA) method on the Weizmann action data set. 
