function feat = extract_feature_dict(img, p0, pos_num, neg_num,opt) %#ok<*FNDEF>
%
opt.negSamples=1;
feat.w=opt.w;
w=opt.w;
patch=opt.patch;
negSamples=opt.negSamples;
if(~exist('negSamples','var'))
negSamples=0;
end
% p = [(bbox(2)+bbox(4))/2, (bbox(1)+bbox(3))/2, bbox(4)-bbox(2), bbox(3)-bbox(1), 0];
% % psize =  trackpars.nsize;
psize = opt.tmplsize;
% p0 = [p(1), p(2), p(3)/psize(1), p(5), p(4)/p(3), 0]'; %param0 = [px, py, sc, th,ratio,phi];   
% p0 = affparam2mat(p0); 
% p0=bbox; % If sent directly affine parameters instead of bounding box
best=p0;
theta = [1,1,0,0,0,0];
gfrm = double(img);
bbox = [best(1), best(2), best(3)*psize(2), best(3)*psize(2)];
%%--- postive sample ---%%
centers = repmat(affparam2geom(p0), [1, pos_num]);
locs = centers + randn(6,pos_num) .* repmat(theta(:), [1, pos_num]);
wimgs = warpimg(gfrm, affparam2mat(locs), psize);                  
feat.feaArr = [];


for i = 1:pos_num
    sample = wimgs(:,:,i);
    
    %% raw gray vales
    %  feat.feaArr = [feat.feaArr sample(:)];
    
    %% hog features
%     hog = double(vl_hog(im2single(sample), 8));
%     hog=LBPTest(sample);  % For Local binary pattern test
%     feat.feaArr = [feat.feaArr hog(:)];
%     hog=im2col(sample,[3 3]);
if(patch)

if(opt.mypatches==1)
size_img = size(img);
size_patch=[w w];
 size_skip = [3 3]; 
 border = 'off';
[patches] = im2patch(sample, size_patch, size_skip, border);
else
    [patches,~,dif_m,dif_n] = im2patches(sample,w);
end
    hog=patches;
else
    hog=sample;
    
end
if(negSamples)
feat.label = [ones(pos_num, 1); -1*ones(neg_num, 1)]';
else
   feat.label = ones(pos_num, 1); 
end
if(opt.MD)
    feat.feaArr=[feat.feaArr ;hog];
else
feat.feaArr = [feat.feaArr , hog(:)];
end
end
%%--- negtive sample ---%%
width= bbox(4) - bbox(2);
height = bbox(3) - bbox(1);
width=bbox(3);
height=bbox(4);

%%  This use if negative samples also being used
if (negSamples)
    overlap_ratio = .05;
cx = [width*(1-overlap_ratio) width*(1+overlap_ratio)];
cy = [height*(1-overlap_ratio) height*(1+overlap_ratio)];
cx=abs(cx);
cy=abs(cy);
centers(1,:) = centers(1,:) + random('uniform',cx(1),cx(2),1,neg_num) .* sign(rand(1,neg_num)-0.5);
centers(2,:) = centers(2,:) + random('uniform',cy(1),cy(2),1,neg_num) .* sign(rand(1,neg_num)-0.5);
wimgs = warpimg(gfrm, affparam2mat(centers), psize);
for i = 1:neg_num
    sample = wimgs(:,:,i);
    
    %% raw gray vales
    %  feat.feaArr = [feat.feaArr sample(:)];
    
    %% hog features
%     hog = double(vl_hog(im2single(sample), 8));
%     hog=LBPTest(sample);  % For Local binary pattern test
%     feat.feaArr = [feat.feaArr hog(:)];
%     hog=im2col(sample,[3 3]);
if(patch)
    if(opt.mypatches==1)
        [patches] = im2patch(sample, size_patch, size_skip, border);
    else
[patches,~,dif_m,dif_n] = im2patches(sample,w);
    end
    hog=patches;
else
    hog=sample;
    
end
if(opt.MD)
    feat.feaArr=[feat.feaArr ;hog];
else
feat.feaArr = [feat.feaArr ,hog(:)];
end
end
end
% feat.feaArr=im2double(feat.feaArr); % For LBP only
% A_norm = sqrt(sum(feat.feaArr .* feat.feaArr));
% feat.feaArr = feat.feaArr ./ (ones(size(feat.feaArr,1),1) * A_norm + eps);
if(patch)  
    if(opt.mypatches==0)
feat.dif_m=dif_m;
   feat.dif_n=dif_n;
    end
   feat.sizeSample=size(sample); % size of sample image (Original block size in image domain)
   [feat.pm feat.pn]=size(hog);
   feat.sizePI=size(hog); % size of patched image (feature extracted image)
end
end
