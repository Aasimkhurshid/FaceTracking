function [patches,pre_labels,dif_m,dif_n] = im2patches(img,w)

[m,n] = size(img);

mm = floor(m/w)*w;
nn = floor(n/w)*w;

iimg = img(1:mm,1:nn);
patches = im2col(iimg,[w w],'distinct');
% patches = im2col(iimg,[w w],'sliding');
pre_labels = -1*ones(1,size(patches,2));
pre_labels(sum(patches)==0) = 0;

dif_m = m - mm;
dif_n = n - nn;




