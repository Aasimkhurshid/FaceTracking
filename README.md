# FaceTracking
Face tracking and Facial landmark tracking
Thank you very much for downloading the code. The code is compatable to run on Matlab 2014a and newere versions. 
For older versions you might need to make some changes.
If you use this code, please cite the paper 
bibtex
"@INPROCEEDINGS{8409614, 
author={A. {Khurshid} and J. {Scharcanski}}, 
booktitle={2018 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)}, 
title={Incremental multi-model dictionary learning for face tracking}, 
year={2018}, 
volume={}, 
number={}, 
pages={1-6}, 
keywords={face recognition;image classification;image enhancement;image reconstruction;learning (artificial intelligence);signal processing;target tracking;face tracking;classification dictionary;face expressions;incremental multimodel dictionary learning;reconstruction dictionary;Face;Dictionaries;Target tracking;Machine learning;Sparse matrices;Lighting;Face Detection;Face Tracking;Dictionary Learning;Incremental Learning;Motion Modeling}, 
doi={10.1109/I2MTC.2018.8409614}, 
ISSN={}, 
month={May},}".
1. This code is freely available be used for research purposes.
2. Please ask for permission if you are using for commercial applications to akhurshid@inf.ufrgs.br or jacobs@inf.ufrgs.br.
3. THERE are THREE BRANCHES, DOWNLOAD ALL THE BRACNHES AND KEEP all three Bracnhes in a FOLDER with each Brach having ITS OWN FOLDER.

HOW TO USE:
1. Please look for folder EXtended functions and install the ksvdbox13 and ompbox10 packages. 
They are used for dictionary leaning and sparse representation. 
For how to install, please refer to their respective readme files.
2. After the installation is complete, Just run the "Demo_MMDL_FaceTracking.m". 
3. Data should be in the "data" folder. If you wish to change how and where to readdata. Please look for "ReadData.m" in "Extended functions" folder.
3. Results will be stored in the "Results" folder. 
TIP: TO improve the code, please feel free to change the parameter values. 
Also, look for the file "runtrackerDictionaries_linux".  It works both for "windows" or "Linux" operating system.To understand Motion modeling
Tracked target face selection, Please refer to the "estwarp_condens_Dict.m".

For some tracking videos using MMDL-FT method, please visit and subscribe "https://www.youtube.com/channel/UCpENSIr2krUrYMhGi73VrEw?view_as=subscriber".

For any questions about code or paper please feel free to contact "akhurshid@inf.ufrgs.br".

