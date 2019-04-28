% clear all; close all; clc; tic;
function [D,output,mu,n,tmplpca]=testDictionaries(wimgs, U0, D0,mu0,n0, ff,OrgsampWimgs,tmplpca,opt)
% %  INPUT
% U0 is old basis tmpl.basis
% D0 is old coeficients/Eigvecs in PCA case
% mu0 is the older mean
% n0 is the number of sampels
% ff is the foregetting factor
% % OUTPUT
% D is the new dictionary
% Output is structure with output.err and output.coeficients
% mu is the new mean
% n is the number of samples

% paramdict.K = 16;
% paramdict.L = 4;
% paramdict.numIteration = 20;
% paramdict.errorFlag = 0;
% paramdict.preserveDCAtom = 0;
% paramdict.InitializationMethod =  'DataElements';
% paramdict.displayProgress = 1;
[N,n] = size(wimgs);
% [D,output] = KSVD(Y,paramdict);
k = 10;
% params.data =feat;% IN case of LBP
params.Tdata = k;
params.dictsize = n;
params.iternum = 30;
params.memusage = 'high';
params.maxatoms=32;
params.opt=opt;
%

% [D,output,err] = ksvd(params,'');


% X = OMP(D,Y,paramdict.L);

%     mu = mean(Y,2);
if (nargin == 1) 
  if (size(wimgs,2) == 1)
    mu = reshape(wimgs(:), size(mu0));
   params.data = wimgs;
    [D,output,err] = ksvd(params,'');
  else
    mu = mean(wimgs,2);
    wimgs = wimgs - repmat(mu,[1,n]);
    mu = reshape(mu, size(mu0));
    params.data = wimgs;
    [D,output,err] = ksvd(params,'');
  end
else
  if (nargin < 6)  ff = 1.0;  end
  if (nargin < 5)  
      n0 = n;  
  end
  if (nargin >= 4 && isempty(mu0) == false)
    mu1 = mean(wimgs,2);
    wimgs = wimgs - repmat(mu1,[1,n]);

    wimgs = [wimgs, sqrt(n*n0/(n+n0))*(mu0(:)-mu1)];
    mu = reshape((ff*n0*mu0(:) + n*mu1)/(n+ff*n0), size(mu0));
    n = n+ff*n0;
    params.data = wimgs;
    params.tmplpca=tmplpca;
    [D,output,err] = ksvd(params,'');
  end
   
% [D,output] = KSVD(Y,paramdict);

 

%     check using this comment below
%       xtest = pinv(D)*counttest;

end