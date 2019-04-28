% clear all;
% close all;
% clc;
% Import all data 
% convert images into gray and to uint8
% p = [px, py, sx, sy, theta]; The location of the target in the first
% frame.
% px and py are th coordinates of the centre of the box
% sx and sy are the size of the box in the x (width) and y (height)
%   dimensions, before rotation
% theta is the rotation angle of the box
% Include true points of data, i.e, True landmarks
clear data;
clear truepts;
clear testing_label_vector1;
clear tmpl;
clear opt;
clear param;
clear param0;
first=1;
% md=1; % IF male dash
fd=1; %if female dash
% vidName='1-FemaleNoGlasses.avi';
% vidName='2-FemaleNoGlasses.avi';
% vidName='5-FemaleNoGlasses.avi';
if(isunix)
    symb='/';
else
    symb='\';
end
% ptsFileDir='E:\Aasim\Data sets\YawDD dataset\Images\Dash\Dash_Female\LabelswithMin_Dash_Female\';
ptsFileDir=pathLabels;
% vidName='11-MaleGlasses.avi';
% DataFileDir='E:\Aasim\Data sets\YawDD dataset\Images\Dash\Dash_Male\';
% ptsFileDir='E:\Aasim\Data sets\YawDD dataset\Images\Dash\Dash_Male\LabelswithMin_Dash_Male\';
dataFolder=[DataFileDir1,vidName,symb];
imagefiles =dir([dataFolder,'*.png']);

% DataFileDir= 'D:\Aasim\Tracking DataSet\FaceOcc1\';
% ImgFolder=[DataFileDir,'img','\'];
% imagefiles =dir([ImgFolder,'*.jpg']);
% dataFolder=ImgFolder;

nfiles = length(imagefiles);    % Number of files found
currentfilename = imagefiles(1).name;
img=imread([dataFolder,imagefiles(1).name]);
img=rgb2gray(img);
% [h w]=size(img);
% data=zeros(h,w,nfiles);
data(:,:,1)=img;
for i=1:nfiles
    currentfilename = imagefiles(i).name;
    img=imread([dataFolder,imagefiles(i).name]);
    img=rgb2gray(img);
    data(:,:,i)=img;
end

%    truepts1=[307.82 158.574 308.374 175.103 310.716 191.507 314.17 207.723 320.76 222.641 331.489 235.148 345.628 245.071 361.689 252.108 378.225 253.852 393.312 250.802 405.766 242.559 416.908 232.088 424.85 219.423 428.835 204.423 430.964 189.116 432.544 174.104 432.692 159.12 322.702 143.794 331.796 136.009 344.004 133.68 356.348 135.625 367.313 140.617 387.913 140.213 398.605 135.51 409.53 134.033 419.781 136.578 425.981 144.313 378.156 153.402 378.605 163.141 379.197 172.936 379.872 183.035 365.847 192.379 372.203 194.312 378.596 196.064 384.865 194.415 390.431 192.635 336.881 156.344 344.275 152.313 352.766 152.634 359.646 157.984 352.355 159.524 343.874 159.527 392.541 157.71 400.039 152.283 408.002 152.372 414.078 156.42 408.596 159.342 400.679 159.367 353.116 214.559 363.567 209.852 372.195 207.321 378.529 209.038 384.898 207.381 393.451 210.192 402.243 214.076 393.661 220.755 385.69 223.423 378.652 224.041 371.939 223.401 363.37 220.867 357.286 214.654 372.13 213.745 378.545 214.174 385.1 213.654 398.291 214.34 385.105 215.089 378.436 215.728 372.025 215.127 ];
% %    true landmarks to start tracking/Initialization of landmarks
% truepts(:,1)=truepts1(1:2:end);
% truepts(:,2)=truepts1(2:2:end);
% truepts=truepts';
groundTruth=1;
if(groundTruth==1)
    if exist('md','var') && (md==1)
        vidName1=vidName(1:end-4);
    elseif exist('fd','var') && (fd==1)
        vidName1=vidName;
        else 
        vidName1=vidName;
    end
fidtruepts=fopen([ptsFileDir,'GT_',vidName,'.txt']);

filebox=[ptsFileDir,'bbox_updated_tcdcn_',vidName(1:end-4),'.txt'];
dat_cell = textread(filebox, '%s', 'delimiter', ',');
dat_cell_x = cell(numel(dat_cell),4);
dat_cell1 = textread(filebox, '%s', 'delimiter', ',');
dat_cell_x1 = cell(numel(dat_cell),4);
for i = 1:numel(dat_cell)
    C = strsplit(dat_cell{i},' ');
    C1 = strsplit(dat_cell1{i},' ');
    if numel(C) == 5    
        dat_cell_x{i,1} = str2num(C{2});
        dat_cell_x{i,2} = str2num(C{3});
        dat_cell_x{i,3} = str2num(C{4});
        dat_cell_x{i,4} = str2num(C{5});
        
        dat_cell_x1{i,1} = str2num(C1{2});
        dat_cell_x1{i,2} = str2num(C1{3});
        dat_cell_x1{i,3} = str2num(C1{4});
        dat_cell_x1{i,4} = str2num(C1{5});
    end
end
my_mat_x=cell2mat(dat_cell_x);
my_mat_x1=cell2mat(dat_cell_x1);


clear dat_cell_x;
clear C;
nfiles=size(my_mat_x,1);
gettruepts=0;
if(fidtruepts>1)
% Get True Points
for i = 1:nfiles
    filename = fgetl(fidtruepts);
    fpts = fscanf(fidtruepts,'%f',136)+1;
    fgetl(fidtruepts);
    fpts = reshape(fpts,[2 68]);
    truepts(:,:,i)=fpts;
    
    
end
WCLMpts=load([ptsFileDir,'trackptsCLM.mat']);
WCLMpts=WCLMpts.trackptsCLM;
 fclose(fidtruepts);
  clear fid;
else
    ignorethis=0;
    if(ignorethis)
    truepts=zeros(2,68,nfiles);
    disp('Please enter the initialization of the total of 68 landmarks on face, eyebrows,nose,eyes and mouth');
    figure,imshow(data(:,:,1));
    [x,y] = ginput(68);
    truepts(:,:,1)=[x,y]';
   ptsFileDir= pathsGT;
    save([ptsFileDir,'truepts',vidName(1:end-4),'.mat'],'truepts');
    end %End of ignore this
end
  px=my_mat_x(first,1)+my_mat_x(first,3)/2;
py=my_mat_x(first,2)+my_mat_x(first,4)/2;
sx=my_mat_x(first,3);
sy=my_mat_x(first,4);
p=[px,py,sx,sy,-0.02];
param0 = [p(1), p(2), p(3)/32, p(5), p(4)/p(3), 0];
   param0 = affparam2mat(param0);

else
%     truepts=[];
    
    truepts=zeros(2,68,nfiles);
%     disp('Enter 68 Landmarks on face'); p=[px,py,sx,sy,-0.02];
   % Parameters are Translation(x,y) ,Rotation angle(theta), Scale (s), Aspect
% ratio (alpha) and skew direction (si)
   param0 = [p(1), p(2), p(3)/32, p(5), p(4)/p(3), 0];
%    param(1),param(2)=Translation
%    param(3)=Scale
%    param(4)=Rotation angle
%    param(5)=Aspect ratio
%    param(6)=skew direction
%    p(6,n) : [dx dy sc th sr phi]'
%    q(6,n) : [q(1) q(3) q(4); q(2) q(5) q(6)]
   param0 = affparam2mat(param0);
%     figure,imshow(data(:,:,1));
%  [x,y] = ginput(68);  
%  truepts(:,:,1)=[x,y]';
  
  clear x;clear y;
  
  [ BB ] = Detect_FaceVJ( data(:,:,1));
   my_mat_x =BB;
   trkptsPr=[];
   sx=my_mat_x(first,3);
sy=my_mat_x(first,4);
sx=2*sx/3;
sy=2*sy/3;
px=my_mat_x(first,1)+sx;
py=my_mat_x(first,2)+sx;
% px=my_mat_x(first,1)+2*sx;
% py=my_mat_x(first,2)+2*sx;
bbox=[px,py,sx,sy];
%% only if the CLM model is available
% GrayITest=im2double(data(:,:,1));
% options.Iterations=100;
%    trueptsCLM = ReInitializeParamsCLM(trkptsPr, GrayITest,bbox,options,vidName);
%    truepts(1,:,1)=trueptsCLM(1:2:end);
%    truepts(2,:,1)=trueptsCLM(2:2:end);
%    truepts(:,:,2:end) =repmat(truepts(:,:,1),1,1,nfiles-1); 
   ignorethis=1;
    if(ignorethis)
    truepts=zeros(2,68,nfiles);
    disp('Please enter the initialization of the total of 68 landmarks on face, eyebrows,nose,eyes and mouth');
    figure,imshow(data(:,:,1));
    [x,y] = ginput(68);
    truepts(:,:,1)=[x,y]';
   ptsFileDir= pathsGT;
    save([ptsFileDir,'truepts',vidName(1:end-4),'.mat'],'truepts');
    end %End of ignore this
    
end


%% Run tracker for 10 times once and then reinitialize 
%   NF=size(data,3);
%   
%      runtracker.m;
%      
%   for idx=10:NF
%   if(mod(NF,10)==0)
%      tie= runtracker(data(:,:,i:i+9),param0);
%         %     pause;
%         %     disp('Reinitialization');
%   end
%   end
%   
  
  % p = [px, py, sx, sy, theta]; The location of the target in the first
% frame.
% px and py are th coordinates of the centre of the box
% sx and sy are the size of the box in the x (width) and y (height)
%   dimensions, before rotation
% theta is the rotation angle of the box
% px=207; py=100;sx=210;sy=195;

%  IN matrix form
%    param(1),param(2)=Translation
%    param(3)=Scale

% subImage=myImage(my_mat_x(1,1):my_mat_x(1,1)+my_mat_x(1,3),my_mat_x(1,2):my_mat_x(1,2)+my_mat_x(1,4));
% subImage1=myImage(my_mat_x(1,2)-20:my_mat_x(1,2)+my_mat_x(1,4)+20,my_mat_x(1,1)-20:my_mat_x(1,1)+my_mat_x(1,3)+20);