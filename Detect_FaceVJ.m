function [ BB ] = Detect_FaceVJ( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
faceDetector=vision.CascadeObjectDetector('ClassificationModel','UpperBody'); %Create a detector object

BB=step(faceDetector,img); % Detect faces
 maxrow=0;
        imax=1;
        for ima=1:size(BB,1)
        if(maxrow<BB(ima,3))
            maxrow=BB(ima,3);
            imax=ima;
        end
        end
    if(size(BB,1)>0)
    BB=BB(imax,:);
    else
        BB=[0 0 0 0];
    end
    
% iimg = insertObjectAnnotation(img, 'rectangle', BB, 'Face'); %Annotate detected faces.
% 
% figure(19);
% imshow(iimg); 
% title('Detected face');
% 
% 
% 
% 
% htextinsface = vision.TextInserter('Text', 'face   : %2d', 'Location',  [5 2],'Font', 'Courier New','FontSize', 14);
% 
% 
% imshow(img);
% hold on
% for i=1:size(BB,1)
%     rectangle('position',BB(i,:),'Linewidth',2,'Linestyle','-','Edgecolor','y');
% end
% hold on
% N=size(BB,1);
% handles.N=N;
% counter=1;
% for i=1:N
%     face=imcrop(img,BB(i,:));
%     savenam = strcat('D:\Detect face\' ,num2str(counter), '.jpg'); %this is where and what your image will be saved
%     baseDir  = 'D:\Detect face\TestDatabase\';
%     %     baseName = 'image_';
%     newName  = [baseDir num2str(counter) '.jpg'];
%     handles.face=face;
%     while exist(newName,'file')
%         counter = counter + 1;
%         newName = [baseDir num2str(counter) '.jpg'];
%     end
% %     fac=imresize(face,[112,92]);
% %     imwrite(fac,newName);
% 
% figure(2);
% imshow(face); 
% title('crop pic');
%    
%     pause(.5);
% 
% end

end

