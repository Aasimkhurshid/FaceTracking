function param = estwarp_condens_Dict(frm,oldfrm, tmpl, param, opt,Dict)
% function param = estwarp_condens(frm, tmpl, param, opt)
% Estimate warping conditions + tracking

%% Copyright (C) .
%% All rights reserved.
% faceDetector = vision.CascadeObjectDetector;
% alpha=0.1; % For ratio between reconstruction and classification
alpha=opt.alpha;
classifDict=0;
dictoutputtest=0;
n = opt.numsample;
sz = size(tmpl.mean);
N = sz(1)*sz(2);

if ~isfield(param,'param')
  param.param = repmat(affparam2geom(param.est(:)), [1,n]);
else
    if(size(param.conf,1)==n)
  cumconf = cumsum(param.conf);
% %   How many particles will be produced is defined by n
% % idx orders the particles according to their probabaility
  idx = floor(sum(repmat(rand(1,n),[n,1]) > repmat(cumconf,[1,n])))+1;
  param.param = param.param(:,idx);
    else
param.param = repmat(affparam2geom(param.est(:)), [1,n]);
param.conf=[];
    end
end
% Generate particles for affine parameters
% Expected image positions #Dynamic Model
% opt.affsig is affine parametes
%  Generate random numbers of specific size
% param.param = param.param + randn(6,n).*repmat(opt.affsig(:),[1,n]);
% For Joint Normal Releigh Directional particle filtering


dir=frm-oldfrm;
if max(dir)==0
    dir=dir+1;
end
y = dir/norm(dir);
 k1=mean(mean(y,2));
 k2=mean(mean(y,1));
% k1=1;
% k2=2;
 stdn=7;
 stdr=7;
 mu=0;
% %  with below 9.2831     573 frames took 155.856 seconds : 3.676ps
%   JNR=sqrt(V) *randn(6,n);
  %  with below 9.2813     573 frames took 134.113 seconds : 4.273ps
%   rng(opt.s);
  JNR=randn(6,n);
%  [ JNR ] = JointNormalReleigh( mu,stdn,stdr,6,n,k1,k2);
% [ JNR ] = drawParticle_DEMC( n );
if(opt.dir==1)
param.param = param.param;
else
param.param = param.param + JNR .*repmat(opt.affsig(:),[1,n]);
end
% [xt1yt1]=param.est(1:2);
% particlespos=param.param(1:2,:);
% particlespos=param.param;
% [xtyt]=mean(particlespos,2);
% particlespos = Directional_Filter(xtyt(1),xtyt(2),xt1yt1(1),xt1yt1(2),particlespos);
% Generate images using particles,
% These are the expected objects, only the object that needs to be tracked.
parampos=affparam2mat(param.param);
targetBoxpre = param.est(:);
  p = affparam2mat(param.param);
  showboxes=0;
  if(showboxes==1)
  figure(5),imshow(frm)
  hold on
 
Int = randi([1 800],1,30);
  for i = 1:size(Int,2)
    drawbox(sz, p(:,Int(i)), 'Color','g');
  end
 drawbox(sz, targetBoxpre, 'Color','r','LineWidth',3);
hold off
  end
wimgs = warpimg(frm,parampos , sz);
%  diff is the deviation from the mean i.e, variance #Observation Model
% bboxes = step(faceDetector, frm);
% crfc=frm(bboxes(2):bboxes(2)+bboxes(4),bboxes(1):bboxes(1)+bboxes(3));
% crfc=imresize(crfc,[32 32]);
% figure(19),imshow(crfc);
% diff = repmat(crfc(:),[1,n]) - reshape(wimgs,[N,n]);
diff = repmat(tmpl.mean(:),[1,n]) - reshape(wimgs,[N,n]);
coefdiff = 0;
if (size(tmpl.basis,2) > 0)
%     Coef is not the eigenvalues for every wimg
if(opt.patch)
patches=[];
for itp=1:size(diff,2)
  img=reshape(diff(:,itp),opt.tmplsize);  

  if(opt.mypatches==1)
                
                size_patch=[opt.w opt.w];
                size_skip=[3 3];
                border='off';
                [patch] = im2patch(img, size_patch, size_skip, border);

            else
               [patch,~,dif_m,dif_n] = im2patches(img,opt.w);
            end
patches=[patches,patch(:)];
end
ptcsz=size(patch);
diff=patches;
clear itp;
end
if(dictoutputtest)
    szz=size(Dict.output,1);
output=Dict.output(1:szz,1:szz);
dictcoef=tmpl.basis*output;
  coef = dictcoef'*diff;
else
  coef = tmpl.basis'*diff;
end
  diff1=diff;
  diff = diff - tmpl.basis*coef;
%   if (isfield(param,'coef'))
%     coefdiff = (abs(coef)-abs(param.coef))*tmpl.reseig./repmat(tmpl.eigval,[1,n]);
%   else
%     coefdiff = coef .* tmpl.reseig ./ repmat(tmpl.eigval,[1,n]);
%   end
  param.coef = coef;
end
%% For Dictionary classification problem
% pos_num=100;neg_num=100;
% feat = extract_feature_dict(frm, param.est(:), pos_num, neg_num);
% Dict.D=[];
%%  Dictionary for classification purpose
if (~isempty(Dict.D))
D=Dict.D;
x=diff1;
trainingData=param.data;
y = D'*trainingData;
y1=D'*diff1;
np = Dict.Psize;
nn =  Dict.Nsize;
% nn = size(Dn,2);
y=y';
y1=y1';

zp = y1(:,1:np);
zn = y1(:,np+1:end);
% if numel(zp(zp~=0)) > numel(zn(zn~=0)) % can use this method as well
Tsum=zp+zn;
%Linear classification
group=[ones(1,np)', 2*ones(1,np)'];
group=group(:);
Wold=y*group;
sparsprod=y*y';
ident=eye(size(sparsprod));
denom=sparsprod+ident;
indenom=inv(denom);
W=indenom*Wold;
sparsesample=D'*x;
cls_err=sparsesample'*W;
% training=W;
%  [class, cls_err] = classify(sample,training,group);
cls_err1=exp(-sqrt(sum(Tsum.^2,2)));
% zpn=exp(-sqrt(sum(zp.^2,2)));
% znn=exp(-sqrt(sum(zn.^2,2)));
% % if numel(zp(zp~=0)) > numel(zn(zn~=0)) % can use this method as well
% Tsum=zpn+znn;
% cls_err=Tsum;
% if sum(zp(zp~=0)) > sum(zn(zn~=0))
% disp('POSITIVE');
% else
% disp('NEGATIVE');
% end
if (~isfield(opt,'errfunc'))  opt.errfunc = [];  end
switch (opt.errfunc)
  case 'robust';
    param.conf = exp(-sum(diff.^2./(diff.^2+opt.rsig.^2))./opt.condenssig)';
  case 'ppca';
    param.conf = exp(-(sum(diff.^2) + sum(coefdiff.^2))./opt.condenssig)';
  otherwise;
%       when it goes wrong, param.conf with method below has NaN
%     param.conf = exp(-sum(diff.^2)./opt.condenssig)';
%      param.conf = exp(-sum(diff.^2))';
        param.conf = exp(-sqrt(sum(diff.^2)))';
end
param.conf = param.conf ./ sum(param.conf);
recerrAll=exp(-sqrt(sum(diff.^2,1)));
rec_err=recerrAll';
%  Find the maximum confidence/Probablility and assign that to image and
% affine parameters
% and similarly error 
% Reconstrut the image using the best probablity given image and error.
if(opt.patch)
patches=[];
for itp=1:size(diff,2) 
    patch=diff(:,itp);
    patch=reshape(patch,ptcsz);
    if(opt.mypatches==1)
        size_img=[opt.tmplsize(1),opt.tmplsize(2)];
        size_patch=[opt.w opt.w];
        [tmpl.mean] = patch2im(patch, size_img, size_patch, size_skip, border);
        
    else
       
        img = patches2im(patch,opt.w,opt.tmplsize(1),opt.tmplsize(2),dif_m,dif_n);
    end
patches=[patches,img(:)];
end
diff=patches;
clear itp;
end

Neff= 1/sum(exp(sum(diff.^2)));
% rec_err=exp(-sum(param.err(:).^2));
% predictValue = alpha * rec_err/sum(rec_err) + (1-alpha) * cls_err/sum(cls_err);
predictValue = alpha * param.conf/sum(param.conf) + (1-alpha) * cls_err/sum(cls_err);

[maxprob,maxidx] = max(predictValue);
rec_err1=rec_err(maxidx);
param.est = affparam2mat(param.param(:,maxidx));
param.wimg = wimgs(:,:,maxidx);
param.err = reshape(diff(:,maxidx), sz);
param.recon = param.wimg + param.err;
param.allwins=reshape(wimgs,[N,n]);
param.parampos=parampos;
param.maxidx=maxidx;
param.cls_err=[param.cls_err,max(cls_err)];
% param.reconerr=[param.reconerr,max(rec_err)];
param.maxprob=maxprob;

else
if (~isfield(opt,'errfunc'))  opt.errfunc = [];  end
switch (opt.errfunc)
  case 'robust';
    param.conf = exp(-sum(diff.^2./(diff.^2+opt.rsig.^2))./opt.condenssig)';
  case 'ppca';
    param.conf = exp(-(sum(diff.^2) + sum(coefdiff.^2))./opt.condenssig)';
  otherwise;
%       when it goes wrong, param.conf with method below has NaN
%     param.conf = exp(-sum(diff.^2)./opt.condenssig)';
%      param.conf = exp(-sum(diff.^2))';
     param.conf = exp(-sqrt(sum(diff.^2)))';
end
param.conf = param.conf ./ sum(param.conf);
%  Find the maximum confidence/Probablility and assign that to image and
% affine parameters
% and similarly error 
% Reconstrut the image using the best probablity given image and error.
predictValue=param.conf;
[maxprob,maxidx] = max(param.conf);
param.est = affparam2mat(param.param(:,maxidx));
param.wimg = wimgs(:,:,maxidx);
param.err = reshape(diff(:,maxidx), sz);
param.recon = param.wimg + param.err;
param.allwins=reshape(wimgs,[N,n]);
param.parampos=parampos;
param.maxidx=maxidx;
param.maxprob=maxprob;
rec_err1=exp(-sum(param.err(:).^2));
% param.reconerr=[param.reconerr,max(rec_err)];
end
param.reconerr=[param.reconerr,rec_err1];
% param.reconerr=[param.reconerr,max(rec_err)];
if(opt.dir==1)
   param.param = resample_particles(param.param, predictValue');
   show_particles( param.param, frm); 
end

end
%% %
%  These particles are drwan the following way.
% First generate n(number of particles you want) affine parameters using normal distribution.
% Next Generate images using these particles, which is done by warping current frame
%  to all these affine parameters, which means that it is drawing 400
%  images which are similiar to the current frame.
% After this conditions are being tested that how far this is from the mean
% of PCA.
% mean of the PCA is also updated.