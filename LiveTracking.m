clear all; %% clear all privious variables
clc;
%warning('off','all'); %.... diable warining msg ...;
camvid = webcam();
% set(camvid,'Resolution','640×480');
I = snapshot(camvid);
% vid = videoinput('UVC Camera (046d:08c5)',1);
% set(vid, 'FramesPerTrigger', Inf);
% set(vid, 'ReturnedColorspace', 'rgb');
% vid.FrameRate =30;
% vid.FrameGrabInterval = 1;  % distance between captured frames 
% start(camvid)
for iFrame =1:100
I=snapshot(camvid);
%imshow(I);
  F = im2frame(I);   
  data(:,:,iFrame)=F;    
end
data1=zeros(480,640,100);
for i=1:size(data,3)

    data1(:,:,i)=rgb2gray(image.cdata);
end
first=1;
aviObject = VideoWriter('myVideo.avi');   % Create a new AVI file
for iFrame = 1:2                   % Capture 100 frames
  % ...
  % You would capture a single image I from your webcam here
  % ...

%   I=snapshot(camvid);
%imshow(I);
%   F = im2frame(I);   
%   data(:,:,iFrame)=F;
 if(iFrame==1)
  
  [ BB ] = Detect_FaceVJ( data(:,:,iFrame));
   my_mat_x =BB;
   trkptsPr=[];
   sx=my_mat_x(first,3);
sy=my_mat_x(first,4);
sx=2*sx/3;
sy=2*sy/3;
px=my_mat_x(first,1)+sx;
py=my_mat_x(first,2)+sx;
bbox=[px,py,sx,sy];
GrayITest=im2double(data(:,:,1));
p=[px,py,sx,sy,-0.02];
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
 else
     vidName='myVideo.avi';
     alphaa=0.8;
     batchsize=3;
     patchsize=8;
     truepts=[];
     first=1;
     ErrorFileID=fopen('errorfileTest','w');
%      runtracker;
  [dispstr,dispstr1]=runtrackerDictionaries_linux(vidName,alphaa,batchsize,patchsize,data,truepts,param0,first,my_mat_x,ErrorFileID,opt);
 end
  
  % Convert I to a movie frame
%   aviObject = addframe(aviObject,F);
  % Add the frame to the AVI file
end
aviObject = close(aviObject);         % Close the AVI file
stop(vid);