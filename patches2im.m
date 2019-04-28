function [img] = patches2im(patches,w,m,n,dif_m,dif_n)

mm = m-dif_m;
nn = n-dif_n;

img_temp = col2im(patches,[w w],[mm nn],'distinct');
% img_temp = col2im(patches,[w w],[mm nn],'sliding');


img = zeros(m,n);
img(1:mm,1:nn) = img_temp;
    

