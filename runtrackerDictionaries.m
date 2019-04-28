function [dispstr,dispstr1]=runtrackerDictionaries(vidNameW,alphaa,batchsize,patchsize,data,truepts,param0,first,my_mat_x,ErrorFileID,opt)
% script: runtrackerDictionaries.m
% requires:
%   data(h,w,nf)
%   param0
%   opt.tmplsize [h,w]
%      .numsample
%      .affsig [6,1]
%      .condenssig

%% Copyright (C) CHECK FOR MEX FILES AND THEN EXECUTE SCRIPT
%% All rights reserved.
% Yellow : True points
% Red: Tracked points
clear iou;
% opt.dump=0; % If do not want to save results
opt.dump=1;  % If want to save Results
% alphaa=0.8;   %For classification and reconstruction tradeoff
opt.alpha=alphaa;
opt.showpart=0; % If dont want to show particles
% opt.showpart=1; % If want to show particles
close all;
if(isunix)
    symb='/';
else
    symb='\';
end
DataFileDir='/media/aasim/E1/Aasim/Data sets/YawDD dataset/Test Videos_Tracking2016/Results/I2MTC Results';
pathtoSaveResults=[DataFileDir,symb,vidNameW];
if(~exist(pathtoSaveResults,'dir'))
mkdir(pathtoSaveResults);
end
pathtoSaveResults=[pathtoSaveResults,symb];
flabel=fopen([vidNameW, '.txt'],'w');
programpath='/home/aasim/Dropbox/Drivers_safety';
errorfile=fopen([vidNameW,'.txt'],'w');
% clc;
opt.s=rng(0,'twister'); %To generate consistant random numbers everytime
numwrong=0;
isGood=1; % To check if the tracked image is good to add in dictionary
% first=300;
tsize=8; % Font size for display text on video and result images
% If use patch based dictionaries, use opt.patch=1, otherwise opt.patch=0
opt.patch=1;  opt.w=patchsize;  %opt.w is the patch size
% opt.mypatches=0; %Distinct patches
opt.mypatches=1; %overlapping patches
% patchsize=4;
opt.isincpca=1; % is 1 if wants to run incremental PCA in KSVD
opt.exactsvd=1; % is 1 if runs exactly svd for ksvd, 0 if runs linear combination instead
% opt.dir=1; % For directional filtering and resampling
opt.dir=0; % For not using directional filtering and resampling
opt.MD=0; %If do not use multiple dictionaries
% opt.MD=1; If want to use multiple dictionaries
% gotill=300;
% gotill=size(data,3); % Number of frames to track
opt.batchsize=batchsize;
if(opt.drivervid)
    gotill=size(my_mat_x,1); % Number of frames to track
else
    gotill=size(data,3);
end

trainingDataP=[];
trainingDataN=[];
trainingDataAll=[];
allCoeffull=[];
CoefofSelected=[];
meanerr(1)=0;
count=0;
mycount=0;
drawsurf=0; %If want to draw surface plot for coeficient visualization
datetoday='';
% updMethod='CLM';
updMethod='DCN';
% updMethod=' ';
TrackingMethod='Dictionaries'; %If using dictionaries as basis to represent data
% vidNameW=['out',TrackingMethod,vidName(1:end-4)];
% pathtoSaveResults=['C:\Users\Aasim\Dropbox\Drivers_safety\Incremental learning for robust visual tracking\ivt\Results\',vidNameW];
% initialize variables
rand('state',0);  randn('state',0);
% first=1;
frame = double(data(:,:,first))/256;
oldframe = double(data(:,:,first))/256;
feature=[];
feature.feaArr = [];
feature.label=[];


if ~exist('opt','var')  opt = [];  end
if ~isfield(opt,'tmplsize')   opt.tmplsize = [32 32];  end
if ~isfield(opt,'numsample')  opt.numsample = 4000;  end
if ~isfield(opt,'affsig')     opt.affsig = [11,9,.05,.05,0,0];  end
if ~isfield(opt,'condenssig') opt.condenssig = 0.01;  end

if ~isfield(opt,'maxbasis')   opt.maxbasis = 16;  end
if ~isfield(opt,'batchsize')  opt.batchsize = 3;  end
if ~isfield(opt,'errfunc')    opt.errfunc = 'L2';  end
if ~isfield(opt,'ff')         opt.ff = 0.95;  end
if ~isfield(opt,'minopt')
    opt.minopt = optimset; opt.minopt.MaxIter = 25; opt.minopt.Display='off';
end

if (isfield(opt,'dump') && opt.dump > 0)
    status=mkdir(vidNameW);
%     cd(vidNameW);
    v=VideoWriter([pathtoSaveResults,vidNameW,'.avi']);
    v1=VideoWriter([pathtoSaveResults,'particles.avi']);
    open(v);
    open(v1);
%     pathtoSaveResults=pwd;
%     cd (programpath)
    % v1=VideoWriter('surface_plot_limitsfirst_component_positive1.avi');
    % open(v1);
    % v2=VideoWriter('surfaceplotlimitsfirstcomponentabsolute1.avi');
    % open(v2);
    % v3=VideoWriter('3 Face Windows Reconstruction First Component1.avi');
    % open(v3);
    % v4=VideoWriter('3 Face Windows Reconstruction First two6 Component1.avi');
    % open(v4);
    fprintf(flabel,num2str(1),'.png');
    fprintf(flabel,'\n');
    mypts=truepts(:,:,1);
    mypts=mypts(:)';
    fprintf(flabel,'%f\t', mypts);
    fprintf(flabel,'\n');
end
mycolor= distinguishable_colors(opt.batchsize);
s1=50;
mysize=repmat(s1,[1 20]);
tmpl.mean = warpimg(frame, param0, opt.tmplsize);
%
sz = size(tmpl.mean);  N = sz(1)*sz(2);
param = [];
param.est = param0;
param.wimg = tmpl.mean;
param.cls_err=[];
param.reconerr=[];
% Here goes dictionary parameteres
Dict=[];
Dict.D=[];
Dict.label=[];
tmpl.basis = [];
tmpl.eigval = [];
tmpl.numsample = 0;
tmpl.reseig = 0;
% For Negative Samples
tmplN=[];
tmplN.basis=[];
tmplN.numsample=0;
tmplN.eigval=[];
% For mean image
pos_num=10;neg_num=10;
feat = extract_feature_dict(frame, param.est(:), pos_num, neg_num,opt);
feature.feaArr=[feature.feaArr,feat.feaArr];
feature.label=[feature.label,feat.label];
col_idsP = find(feature.label == 1);
trainingDataP=feature.feaArr(:,col_idsP);
col_idsN = find(feature.label == -1);
trainingDataN=feature.feaArr(:,col_idsN);
% tmpl.mean=reshape(mean(trainingDataP,2),[32,32]); %does not work so well
if(opt.patch)
    tmpmean=mean(trainingDataN,2);
    tmpmean=reshape(tmpmean,[feat.pm , feat.pn]);
        
    size_img=[feat.sizeSample(1),feat.sizeSample(2)];
    size_patch=[feat.w feat.w];
    size_skip=[3 3];
    border = 'off';
    if(opt.mypatches==1)
    [tmplN.mean] = patch2im(tmpmean, size_img, size_patch, size_skip, border);
    else
        [tmplN.mean] = patches2im( tmpmean,feat.w,feat.sizeSample(1),feat.sizeSample(2),feat.dif_m,feat.dif_n);
    end
    clear  tmpmean;
else
    tmplN.mean=reshape(mean(trainingDataN,2), opt.tmplsize);
end
feature=[];
feature.feaArr = [];
feature.label=[];
trainingDataN=[];
trainingDataP=[];
%% For LDA BEGINS
% paramLDA = [];
% tmplLDA.mean = warpimg(frame, param0, opt.tmplsize);
% [trainingData,labelsTraining ] = ManageLDA( frame,my_mat_x );
% tmplLDA.meanN=trainingData(2,:);
% paramLDA.wimg=trainingData(1,:);
% paramLDA.Nimg=trainingData(2,:);
% paramLDA.est = param0;
%
%   [ paramLDA.A ,paramLDA.T, paramLDA.G ] = directlda(trainingData,labelsTraining);
%   paramLDA = estwarp_condensLDA(frame, tmplLDA, paramLDA, opt,trainingData);
% display_pts(trainingData,labelsTraining,paramLDA.A);

%%  LDA ENDS
%  totaldiff=zeros(size(data,3),1);
% checking true position of points if they exist, for tracking
% initialization
% If you want to use your own points, just initialize 'pts' variable with
% your initialization points
%%
if (exist('truepts','var') && truepts(1)>0)
    npts = size(truepts,2);
    opt.npts=npts;
    aff0 = affparaminv(param.est);
    pts0 = aff0([3,4,1;5,6,2]) * [truepts(:,:,first); ones(1,npts)];
    pts = cat(3, pts0 + repmat(sz'/2,[1,npts]), truepts(:,:,first));
    trackpts = zeros(size(truepts));
    trackerr = zeros(1,gotill); meanerr = zeros(1,gotill);
    ErrorCenter=zeros(1,gotill); averErrorCenter=zeros(1,gotill);
    totaldiffBasis = zeros(1,gotill); trkptsdiff = zeros(1,gotill);
    filteredTrkptsErr= zeros(1,gotill); filteredTrkptsDiff= zeros(1,gotill);
    reconerrwith1=zeros(1,gotill); error_per_image=zeros(1,gotill); 
else
    pts = [];
    ErrorCenter=zeros(1,gotill); averErrorCenter=zeros(1,gotill);
    zeros(1,gotill);

end
%%
% draw initial track window
tfig= figure(1);
%        set(tfig,'rend','painters','pos',[0 0 900 600]) 
drawopt = drawtrackresult([], 0, frame, tmpl, param, pts);
disp('resize the window as necessary, then press any key..');
% pause;
drawopt.showcondens = 0;  drawopt.thcondens = 1/opt.numsample;
wimgs = [];
if (isfield(opt,'dump') && opt.dump > 0)
    %   imwrite(frame2im(getframe(gcf)),sprintf('dump/%s.0000.png',title));
    %   save(sprintf('dump/opt.%s.mat',title),'opt');
%     cd (vidNameW)
        imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,num2str(1),'.png']);
%         cd (programpath)
    writeVideo(v,frame2im(getframe(gcf)));
end
tmplpcap=tmpl;
tmplpcan=tmplN;

%% track the sequence from frame 2 onward
load basis1;
f1=1;

duration = 0; tic;
theta=zeros(gotill,1);
%  totaldiff=zeros(gotill,1);
firstIteration = true;
flag=0;
if (exist('dispstr','var'))  dispstr='';  end
for f = first:gotill
    %     size(data,3)
    %check if ReInitialization of Tracking landmarks is required
    %     if(f==300)
    % %         opt.numsample=5000;
    % %         opt.affsig = [20,10,.02,.02,.005,.001];
    % %         opt.batchsize=1;
    % pause;
    %     end
    
    
    
    
    
    
    
    
    frame = double(data(:,:,f))/256;
    if f>1
        oldframe=double(data(:,:,f-1))/256;
    end
    % do tracking
    % Estimate warping conditions + tracking
    %  param = estwarp_grad(frame, tmpl, param, opt);
    param = estwarp_condens_Dict(frame,oldframe, tmpl, param, opt,Dict);
    
    %% Aasim update Begin
    % For LDA
    % [trainingData,labelsTraining ] = ManageLDA( frame,my_mat_x,trainingData,labelsTraining );
    %   [ paramLDA.A ,paramLDA.T, paramLDA.G ] = directlda(trainingData,labelsTraining);
    %   paramLDA = estwarp_condensLDA(frame, tmplLDA, paramLDA, opt,trainingData);
    % display_pts(trainingData,labelsTraining,paramLDA.A);
    
    %% Aasim Udate End
    % do update
    if(f>3)
      isGood = update_check_Dict(param.reconerr(f-1),param.reconerr, param.maxidx);
    end
    % [ theta1,thetaVec,totaldiffBasistemp,isGood ] = checkUpdateReq( basis1,tmplbasis ,f,trkptsdiff(f-1),filteredTrkptsDiff(f-1));
    % check something with basis, and also particles
    isGood=1;
    if(isGood)
        wimgs = [wimgs, param.wimg(:)];
        %     trainingDataAll=[trainingDataAll,param.wimg(:)];
        pos_num=10;neg_num=10;
        feat = extract_feature_dict(frame, param.est(:), pos_num, neg_num,opt);
        feature.feaArr=[feature.feaArr,feat.feaArr];
        feature.label=[feature.label,feat.label];
        col_idsP = find(feature.label == 1);
        trainingDataP=feature.feaArr(:,col_idsP);
        col_idsN = find(feature.label == -1);
        trainingDataN=feature.feaArr(:,col_idsN);
    else
%         isGood=1;
        numwrong=numwrong+1;
        
    end
    % trainingDataPos=[trainingDataP,trainingDataAll];
    
    if (size(wimgs,2) >= opt.batchsize) % Update dictionary or not
                 


        if(opt.patch)
            if(opt.mypatches==1)
                [tmpl.mean] = im2patch(tmpl.mean, size_patch, size_skip, border);
                [tmplN.mean] = im2patch(tmplN.mean, size_patch, size_skip, border);
            else
                [tmpl.mean,~,dif_m,dif_n] = im2patches(tmpl.mean,opt.w);
                [tmplN.mean,~,dif_m,dif_n] = im2patches(tmplN.mean,opt.w);
            end
            
        end
        [Dp,output,mu,tmpl.numsample,tmplpcap]=testDictionaries(trainingDataP, tmpl.basis, tmpl.eigval,tmpl.mean,tmpl.numsample, opt.ff,wimgs,tmplpcap,opt);
        [Dn,outputN,tmplN.mean,tmplN.numsample,tmplpcan]=testDictionaries(trainingDataN, tmplN.basis, tmplN.eigval,tmplN.mean,tmplN.numsample, opt.ff,wimgs,tmplpcan,opt);
        tmplN.basis=Dn;
        Dict.D=[Dp,Dn];
                    param.data=[trainingDataP,trainingDataN];

        if(firstIteration )
            firstIteration = false;
            basis1=tmpl.basis;
            D1=Dp;
        end
    tmplbasis{f}=tmpl.basis;
        Dict.label=feature.label;
        Dict.output=output;
        Dict.Psize=size(Dp,2);
        Dict.Nsize=size(Dn,2);
        
%         tmpl.basis=Dp;
tmpl.basis=D1;
        currentCoeff=tmpl.basis;
        currentScore=tmpl.mean(:);
        if(opt.patch)
            if(opt.mypatches==1)
                size_img=[feat.sizeSample(1),feat.sizeSample(2)];
                size_patch=[feat.w feat.w];
                [tmpl.mean] = patch2im(tmpl.mean, size_img, size_patch, size_skip, border);
                size_img=[feat.sizeSample(1),feat.sizeSample(2)];
                size_patch=[feat.w feat.w];
                [tmplN.mean] = patch2im(tmplN.mean, size_img, size_patch, size_skip, border);
            else
                [tmpl.mean] = patches2im( tmpl.mean,feat.w,feat.sizeSample(1),feat.sizeSample(2),feat.dif_m,feat.dif_n);
                [tmplN.mean] = patches2im( tmplN.mean,feat.w,feat.sizeSample(1),feat.sizeSample(2),feat.dif_m,feat.dif_n);
            end
            
        else
            tmpl.mean=reshape(mu, opt.tmplsize);
        end
        %         tmpl.eigval= output.CoefMatrix;
        %   figure(6),
        % scatter(tmpl.basis(:,1),tmpl.basis(:,2),'MarkerEdgeColor','b'),ylim([-0.4 0.4]),xlim([-0.4 0.4]);
        
        
        
        % hold on
        % scatter(tmpl.mean(:),'MarkerEdgeColor','r');
        % hold off
        %     figure,plot(basis1);
        % Old Method
        wimgs=[];
        clear feature;
        feature=[];
        feature.feaArr = [];
        feature.label=[];
        trainingDataP=[];
        trainingDataN=[];
        % % % % % % % % % % % % % % % Aasim updated
        %  if size(wimgs,2)>5
        % % This way method does not forget all the previous when update in SKLM is
        % % done, which means information is not lost as whole but information is
        % % reduced with time,which gives smooth transition in basis.
        %
        %     wimgs = wimgs(:,2:end);
        %  end
        %     if (size(tmpl.basis,2) > opt.maxbasis) %use this if need coef
        
        if(0)
            %tmpl.reseig = opt.ff^2 * tmpl.reseig + sum(tmpl.eigval(tmpl.maxbasis+1:end).^2);
            tmpl.reseig = opt.ff * tmpl.reseig + sum(tmpl.eigval(opt.maxbasis+1:end));
            tmpl.basis  = tmpl.basis(:,1:opt.maxbasis);
            %       tmpl.eigval = tmpl.eigval(1:opt.maxbasis);
            if (isfield(param,'coef'))
                param.coef = param.coef(1:opt.maxbasis,:);
            end
            %        wimgs = wimgs(:,2:end);
        end
        flagtest=1;
        flag=0; % If want to update, Uncomment this.
        if(flagtest)
        %            Check if Update is required or not
        [ theta1,thetaVec,totaldiffBasistemp,flag ] = checkUpdateReq( basis1,tmplbasis ,f,trkptsdiff(f-1),filteredTrkptsDiff(f-1));
         totaldiffBasis(f)=totaldiffBasistemp;
        end
         if(flag)
            mycount=mycount+1;
%               
%            basis1=tmpl.basis;
%            %     figure(4);  plot(theta(f),'b.-');
% %         if (totaldiff(f)>5)
if(strcmp(updMethod,'DCN'))
    trueptsCLM=truepts(:,:,f);
    BB=[0 0 0 0];
%         while (BB(1)==0)
%             frame = double(data(:,:,f))/256;
%             %    img=imresize(img,[sizeofimage1,sizeofimage2]);
%             [ BB ] = Detect_FaceVJ( img);
%          
% %             f=f+1;
%         end
     height=truepts(2,9,f)-(truepts(2,20,f)+truepts(2,25,f))/2;
width=truepts(1,17,f)-truepts(1,1,f);
px=truepts(1,31,f);
py=truepts(2,31,f);
BB(1)=px;
BB(2)=py;
BB(3)=width;
BB(4)=height;

sx=BB(3);
sy=BB(3);

bbox=[px,py,sx,sy];
else
   
    %             wimgs=[];
    options.Iterations=50;
    if(exist('mymatx','var'))
    BB=my_mat_x(f,:);
           diffsize=BB(3)-my_mat_x(f,3);
    diffsized2=diffsize/2;
     BB(1:2)=BB(1:2)+diffsized2;
    BB(3:4)=my_mat_x(f,3:4);
    else
        trkptsPr=trackpts(:,:,f-1);
         basis1=tmplbasis{f};
        BB=[0 0 0 0];
        while (BB(1)==0)
            frame = double(data(:,:,f))/256;
            %    img=imresize(img,[sizeofimage1,sizeofimage2]);
            [ BB ] = Detect_FaceVJ( img);
            BB(1:2)=BB(1:2)+BB(3)/2;
%             f=f+1;
        end
       
       
    end
    
    
   
    
   

    [trueptsCLMtmp,ErrorCLM] = ReInitializeParamsCLM(trkptsPr,frame,BB,options,vidName,DataFileDir);
    ErrorCLMarray=[ErrorCLMarray,ErrorCLM];
    trueptsCLM(1,:)=trueptsCLMtmp(1:2:end);
    trueptsCLM(2,:)=trueptsCLMtmp(2:2:end);
    clear trueptsCLMtmp;
   
end
                % % mycount=mycount+1;
%                 basis1=tmplbasis{f};
%                 trkpts=trackpts(:,:,f-1);
                %       trueptsCLM=truepts(:,:,f);
                trackpts(:,:,f)=trueptsCLM;
               

               savedata=param.data;
                [tmpl,param,pts0]=reInitializeParams(BB,tmpl,f,frame,trueptsCLM,param,param0,opt);
              param.data=savedata;
          end
        
        
        
        
    end %This ends the if that checks if we can update PCA (SKLM)
    
    %% Test for alternate frame calling of CLM
    %   if(mod(f,2)==0 && f>2 && flag==0)
    %       options.Iterations=3;
    %       trueptsCLMtmp = ReInitializeParamsCLM(trackpts(:,:,f-1),frame,my_mat_x(f,:),options);
    %       trueptsCLM(1,:)=trueptsCLMtmp(1:2:end);
    %       trueptsCLM(2,:)=trueptsCLMtmp(2:2:end);
    %       clear trueptsCLMtmp;
    %       flag=1;
    %       trkpts=trackpts(:,:,f-1);
    %       %       trueptsCLM=truepts(:,:,f);
    %       trackpts(:,:,f)=trueptsCLM;
    %   end
    %% Aasim Edit end for surface plots
    if(isfield(param,'coef'))
        % [CoefofSelected ] =FindCoeff( tmpl.basis,wimgs(:,size(wimgs,2)) );
        [CoefofSelected ]=param.coef(:,param.maxidx)';
        % [CoefAllWinsCurrent]=FindCoeff( tmpl.basis,param.allwins );
        [CoefAllWinsCurrent]=param.coef';
        % % % % allCoeffull=[allCoeffull;allCoef];
        selectedPos=param.parampos(1:2,param.maxidx);
        CoefAllParticles=CoefAllWinsCurrent(:,1);
        if(drawsurf)
            allpos=param.parampos(1:2,:);
            matrixsurface=zeros(size(frame));
            Xpos=allpos(1,:);
            Ypos=allpos(2,:);
            for idx_pos=1:length(Xpos),
                matrixsurface(round(Xpos(idx_pos)),round(Ypos(idx_pos)))=CoefAllParticles(idx_pos);
            end
            
            [dX,dY]=meshgrid(1:size(frame,1),1:size(frame,2));
            figure(9),surf(dX,dY,matrixsurface');
            matrixsurface=matrixsurface';
            selectedpositionCoef=matrixsurface(round(selectedPos(2,1)),round(selectedPos(1,1)));
            % selectedValue=selectedpositionCoef;
            selectedValue=CoefofSelected(:,1);
            % selectedValue=allCoef(1,1);
            mat1=[round(selectedPos(2,1)),round(selectedPos(1,1))];
            mat2=[ceil(min(Ypos))-5,ceil(min(Xpos))-5];
            posInSurf2=mat1-mat2;
            % figure,imshow(matrixsurface,[]);
            matrixsurface2=matrixsurface(ceil(min(Ypos))-5:ceil(min(Ypos))+30,ceil(min(Xpos))-5:ceil(min(Xpos))+30);
            [dX2,dY2]=meshgrid(1:size(matrixsurface2,1),1:size(matrixsurface2,2));
            figure(10),
            surf(dX2,dY2,matrixsurface2);
            % % surf(dX2,dY2,abs(matrixsurface2));
            hold on
            stem3(posInSurf2(1,1),posInSurf2(1,2),selectedValue,'MarkerSize',10,'MarkerEdgeColor','green','MarkerFaceColor','g');
            % stem(posInSurf2(1,1),posInSurf2(1,2),'MarkerSize',10,'MarkerEdgeColor','green','MarkerFaceColor','g');
            xlim([0 40]) ,ylim([0 40]),zlim([-6 6]);
            hold off;
            % writeVideo(v1,frame2im(getframe(gca)));
            % close figure 10;
            figure(11),
            surf(dX2,dY2,abs(matrixsurface2));
            hold on
            stem3(posInSurf2(1,1),posInSurf2(1,2),abs(selectedValue),'MarkerSize',10,'MarkerEdgeColor','green','MarkerFaceColor','g');
            xlim([0 40]) ,ylim([0 40]),zlim([0 6]);
            hold off;
            % writeVideo(v2,frame2im(getframe(gca)));
            % close figure 11
        end
        %%
        % if use three components
        % figure(7),scatter3(allCoef(:,1),allCoef(:,2),allCoef(:,3),100,mycolor(:,1)),title('Components');
        % If use one component
        % figure(7),scatter(allCoef(:,1),ones(size(allCoef,1),1),100,mycolor(:,1)),title('Components');
        % figure(7),stem3(CoefWinsCurrent(:,1),'LineStyle','none','Marker','*');
        % hold on
        % stem3(allCoef(20,1),'LineStyle','none','color','red');
        % figure,plot(CoefWinsCurrent(:,1),CoefWinsCurrent(:,2));
        % plot in time
        % scatter3(CoefWinsCurrent(:,1),CoefWinsCurrent(:,2),ones(size(CoefWinsCurrent,1),1))
        % hold on
        % scatter3(allCoef(1,1),allCoef(1,2),ones(size(1),1),'r')
        % for ttime=1:20
        % scatter3(ttime*ones(size(CoefWinsCurrent,1),1),CoefWinsCurrent(:,1),'b')
        % hold on
        % scatter(ttime*ones(size(1),1),allCoef(ttime,1),'r')
        % end
        % pause;
        % figure(7),stem(allCoef(:,1));
        % title('second Component of PCA against time');
        % xlabel('Time');
        % ylabel('1st Components');
        % zlabel('3rd Components');
        % xlim([-10 10]) ,
        % ylim([-10 10])
        % ,zlim([-5 5])
        
        %% Aasim Edit For Reconstruction using Components Start
        % Video1 Face Windows
        % Video 2 for Face Reconstruction using Component 1
        % Face window reconstruction Error
        % for flg=1:16
        % if (sum(tmpl.eigval(1:flg))/sum(tmpl.eigval)) >0.95
        % % disp(num2str(flg));
        % break
        % end
        % end
        if(opt.patch)
            if(opt.mypatches==1)
                size_img = size(frame);
                size_patch=[opt.w opt.w];
                size_skip = [3 3];
                border = 'off';
                [patchescurrent] = im2patch(param.wimg, size_patch, size_skip, border);
            else
                [patches,~,dif_m,dif_n] = im2patches(sample,w);
            end
        end
        dict1coef=D1'*patchescurrent(:);
        diffwith1=patchescurrent(:)-D1*dict1coef;
        reconerrwith1(f)= exp(-sqrt(sum( diffwith1.^2)))';
        if (exist('drawbasis','var'))
            ReconImageT=reshape((tmpl.basis(:,1:flg)*CoefofSelected(1,1:flg)'+tmpl.mean(:)), opt.tmplsize);
            ReconImage1=reshape((CoefofSelected(1,1)*tmpl.basis(:,1)+tmpl.mean(:)), opt.tmplsize);
            % figure(12),imshow(ReconImage1,[]),title('Reconstruction using first component');
            ReconImage2=reshape((tmpl.basis(:,1:2)*CoefofSelected(1,1:2)'+tmpl.mean(:)), opt.tmplsize);
            % figure(13),imshow(ReconImage2,[]),title('Reconstruction using first two component');
            ErrorFromOurRecon=param.wimg-ReconImage1;
            ErrorFromOurRecon2=param.wimg-ReconImage2;
            ErrorFromOurReconT=param.wimg-ReconImageT;
            err1=sum(sum(abs(ErrorFromOurRecon)));
            err2=sum(sum(abs(ErrorFromOurRecon2)));
            errT=sum(sum(abs(ErrorFromOurReconT)));
            errorg=sum(sum(abs(param.err)));
            figure(14),
            subplot(1,5,1), subimage(param.wimg),title('Face Window')
            subplot(1,5,2), subimage(ReconImage1),title('FirstCompRecon')
            subplot(1,5,3), subimage(param.recon),title('ReconProposed')
            subplot(1,5,4),subimage(ErrorFromOurRecon),title(['Our:',num2str(err1)])
            subplot(1,5,5),subimage(param.err),title(['Org:',num2str(errorg)])
            writeVideo(v3,frame2im(getframe(gcf)));
            figure(15),
            subplot(1,5,1), subimage(param.wimg),title('Face Window')
            subplot(1,5,2), subimage(ReconImageT),title('TwoComp Recon')
            subplot(1,5,3), subimage(param.recon),title('ReconProposed')
            subplot(1,5,4),subimage(ErrorFromOurReconT),title(['Our:',num2str(err2)])
            subplot(1,5,5),subimage(param.err),title(['Org:',num2str(errorg)])
            writeVideo(v4,frame2im(getframe(gcf)));
        end
        %% Aasim Edit For Reconstruction using Components End
    end
    
    
    
    
    duration = duration + toc;
    % Calculate Errors
       %Center Error
       sx1=my_mat_x(f,3);
sy1=my_mat_x(f,4);
% sx1=2*sx1/3;
% sy1=2*sy1/3;
px1=my_mat_x(f,1)+sx1/2;
py1=my_mat_x(f,2)+sy1/2;
param.esttemp=param.est(:);
    ErrorCenter(f) = sqrt(sum( (([px1,py1]-param.esttemp(1:2)').^2),2));
    averErrorCenter(f)= mean( ErrorCenter(~isnan( ErrorCenter)&( ErrorCenter>0)));
%             if (exist('dispstr1','var'))  fprintf(repmat('\b',[1,length(dispstr1)]));  end;

     dispstr1 = sprintf('%d: %.4f / %.4f',f,ErrorCenter(f),averErrorCenter(f));
     if (~(exist('truepts','var') && truepts(1)>0))
       if (exist('dispstr1','var'))  fprintf(repmat('\b',[1,length(dispstr1)]));  end;
        fprintf(dispstr1);
        
     end
        % OverLap Ratio
       overlapratio=0;
if (overlapratio)
% # determine the (x, y)-coordinates of the intersection rectangle
boxA=[px1-3*sx1/2,py1,px1,py1+sy1];
boxB=[bboxB1(1)-3*sx1/2,bboxB1(2),bboxB1(1),bboxB1(2)+sy1];
	xA = max(boxA(1), boxB(1));
	yA = max(boxA(2), boxB(2));
	xB = min(boxA(3), boxB(3));
	yB = min(boxA(4), boxB(4));
 
% 	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1);
 
% 	# compute the area of both the prediction and ground-truth
% 	# rectangles
	boxAArea = (boxA(3) - boxA(1) + 1) * (boxA(4) - boxA(2) + 1);
	boxBArea = (boxB(3) - boxB(1) + 1) * (boxB(4) - boxB(2) + 1);
 
% 	# compute the intersection over union by taking the intersection
% 	# area and dividing it by the sum of prediction + ground-truth
% 	# areas - the interesection area
	iou(f) = interArea / (boxAArea + boxBArea - interArea);
end
    % draw result
    if (exist('truepts','var') && truepts(1)>0)
        %       Aasim Comments
        %       For traking points, param.est must have matching dimensions as
        %       number of points we want to track
        % The tracking of parameters is just applying affine transformation from the
        % face box to all the landmark parameters.
        if(flag==0)
            trackpts(:,:,f) = param.est([3,4,1;5,6,2])*[pts0; ones(1,npts)];
        end
        
        %     truepts(:,:,f) is required, Figure out why????
        %     To display truepts as well to visualize
        % To reinitialize with truepoints after every 10 points
        % Later will be combined by CLM
        
        
        pts = cat(3, pts0+repmat(sz'/2,[1,npts]), truepts(:,:,f),trackpts(:,:,f));
        %     disp('Reinitialization...');
        %     pause;
        
        
        
        idx = find(pts(1,:,2) > 0);
        if (length(idx) > 0)
            % trackerr(f) = mean(sqrt(sum((pts(:,idx,2)-pts(:,idx,3)).^2,1)));
            trackerr(f) = sqrt(mean(sum((pts(:,idx,2)-pts(:,idx,3)).^2,1)));
            idx1=[37,46];
           trackerr1(f) = trackerr(f)/mean(sqrt(sum((pts(:,37,2)-pts(:,46,2)).^2,1)));
           if(opt.drivervid)
           [ error_per_image(f) ] = compute_error( pts(:,idx,2)', pts(:,idx,3)' );
           end
           fprintf(errorfile,'%d \t %f \t %f', f ,trackerr(f),num2str(trackerr(f)));
           fprintf(errorfile,'\n');

            filteredTrkptsErr(f)=median(median(trackerr(1:f),'omitnan'));
            if (f > 2)
                %               trkptsdifftemp=sum(abs(trackpts(:,:,f)-trackpts(:,:,f-1)));
                trkptsdifftemp = sqrt(mean(sum((trackpts(:,:,f)-trackpts(:,:,f-1)).^2)));
                
                trkptsdiff(f)=sum(trkptsdifftemp(:));
                
                filteredTrkptsDiff(f)=median(median(trkptsdiff(1:f),'omitnan'));
            end
            
            
            
        else
            trackerr(f) = nan;
            trkptsdiff(f)=nan;
            filteredTrkptsDiff(f)=nan;
            filteredTrkptsErr(f)=nan;
            error_per_image(f)=nan;

        end
        meanerr(f+1) = mean(trackerr(~isnan(trackerr)&(trackerr>0)));
%         if(meanerr(f)>20)
%             return;
%         end
        if (exist('dispstr','var'))  fprintf(repmat('\b',[1,length(dispstr)]));  end;
        dispstr = sprintf('%d: %.4f / %.4f',f,trackerr(f),meanerr(f));
        dispstr=[dispstr,',',dispstr1];
        fprintf(dispstr);
       
        %     figure(2);  plot(trackerr,'r.-'),title('Tracking Error');
        %     figure(3),plot(trkptsdiff,'b.-'),title('Differnce in Tracking points');
        %     figure(4);  plot(filteredTrkptsErr,'r'),title('Filtered Median incremental Tracking Error');
        %     figure(5),plot( filteredTrkptsDiff,'b'),title('Filtered Median incremental Differnce in Tracking points');
        
       tfig= figure(1);
%        set(tfig,'rend','painters','pos',[0 0 900 600]) 

    end
    
    tmplnew.basis=[];
    if (opt.patch)
        tmplnew.mean=tmpl.mean;
        tmplnew.numsample=tmpl.numsample;
        tmplnew.reseig=tmpl.reseig;
        if(size(tmpl.basis,2)>1)
            for basi=1:size(tmpl.basis,2)
                sample=tmpl.basis(:,basi);
                sample=reshape(sample,feat.sizePI);
                if(opt.mypatches==1)
                    size_img=[feat.sizeSample(1),feat.sizeSample(2)];
                    [samplenew] = patch2im(sample, size_img, size_patch, size_skip, border);
                else
                    samplenew = patches2im(sample,feat.w,feat.sizeSample(1),feat.sizeSample(2),feat.dif_m,feat.dif_n);
                end
                tmplnew.basis=[tmplnew.basis,samplenew(:)];
                tmplnew.eigval=tmpl.eigval;
            end
        else
            tmplnew.basis=tmpl.basis;
            tmplnew.eigval=tmpl.eigval;
        end
        
        drawopt = drawtrackresult(drawopt, f, frame, tmplnew, param, pts);
        clear tmplnew;
        clear basi;
        clear sample;
    else
        drawopt = drawtrackresult(drawopt, f, frame, tmpl, param, pts);
    end
    
    
    %%% UNCOMMENT THIS TO SAVE THE RESULTS (uses a lot of memory)
    % %      saved_params{f} = param;
    if (isfield(opt,'dump') && opt.dump > 0)
%         cd (vidNameW)
        imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,num2str(f),TrackingMethod,'.png']);
%         cd (programpath)
        writeVideo(v,frame2im(getframe(gcf)));
    end
    if (isfield(opt,'showpart') && opt.showpart > 0)
        figure(2),show_particles( param.param, frame);
        if( isfield(opt,'dump') && opt.dump > 0)
            
            writeVideo(v1,frame2im(getframe(gcf)));
        end
    end
    tic;
    
    
end
duration = duration + toc;
fprintf('     %d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);
if(exist('ErrorFileID','var'))
fprintf(ErrorFileID,[ ' , ',num2str(dispstr)]);
fprintf(ErrorFileID,'     %d frames took %.3f seconds : %.3fps\n',f,duration,f/duration);
fprintf(ErrorFileID, ' \n ');
end
% end of function
if(opt.dump)
    close(v);
    close(v1);
    % close(v2);
    % close(v3);
    % close(v4);
    fprintf(flabel,num2str(f),'.png');
    fprintf(flabel,'\n');
    mypts=trackpts(:,:,f);
    mypts=mypts(:)';
    fprintf(flabel,'%f\t', mypts);
    fprintf(flabel,'\n');
end
if (isfield(opt,'dump') && opt.dump > 0)
%     cd(vidNameW)
%     pathtoSaveResults='';
vidName=vidNameW;
    correlation1=corr([trkptsdiff', trackerr']);
    figure,plot( trackerr,'r'),title(['Track Error with Correlation . ',num2str(correlation1(2)),'Mean Error:' num2str(mean(meanerr))],'FontSize',tsize),xlabel('Frame Number'),ylabel('Track Error(red)');
    if (isfield(opt,'dump') && opt.dump > 0)imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidName,datetoday,'PCA_Track Error.png']);end
    figure,plot( trkptsdiff,'b'),title(['Track points Difference with Correlation .',num2str(correlation1(2)),'Mean Error:' num2str(mean(meanerr))],'FontSize',tsize);
    if (isfield(opt,'dump') && opt.dump > 0)imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidName,datetoday,'PCA_Track points Difference.png']);end
    trkptsdiff1=normalizeData(trkptsdiff');
    trackerr1=normalizeData(trackerr');
    figure,plot(trackerr1(1:5:end),'r')
    hold on
    plot(trkptsdiff1(1:5:end),'b'),title([' Track Error Vs Track points difference with Correlation: ',num2str(correlation1(2)),'Mean Error:' num2str(mean(meanerr))],'FontSize',tsize),xlabel('Frame Number'),ylabel('Track Error(red)/Error Predictor(blue)');
    if (isfield(opt,'dump') && opt.dump > 0) imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidName,datetoday,'Track Error Vs Track points differenceCorr_',num2str(correlation1(2)),'.png']);end
    % Filtered Results
    filteredTrkptsDiff1=medfilt1(trkptsdiff1,50);
    filteredTrackErr1=medfilt1(trackerr1,30);
    correlation=corr([filteredTrkptsDiff1, filteredTrackErr1]);
    figure,plot(filteredTrkptsDiff1),title(['Median filtered Track points difference with Correlation .',num2str(correlation(2)),'_Mean Error:',num2str(mean(meanerr))],'FontSize',tsize);
    if (isfield(opt,'dump') && opt.dump > 0) imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidNameW,datetoday,'Median filtered Track points difference.png']);end
    figure,plot( filteredTrackErr1,'r'),title(['Median filtered Track Error with Correlation .',num2str(correlation(2)),'_Mean Error:',num2str(mean(meanerr))],'FontSize',tsize);
    if (isfield(opt,'dump') && opt.dump > 0) imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidNameW,datetoday,'Median filtered Track Error.png']);end
    [filteredTrkptsDiff1 ] = normalizeData( filteredTrkptsDiff1 );
    [filteredTrackErr1] = normalizeData( filteredTrackErr1);
    figure,plot(filteredTrackErr1(1:5:end),'r')
    hold on
    plot(filteredTrkptsDiff1(1:5:end),'b'),title(['Med filt Trck Err Vs Trck pts diff with Corr .',num2str(correlation(2)),'_Mean Error:',num2str(mean(meanerr)), 'ReInit Num:', num2str(mycount)],'FontSize',tsize),xlabel('Frame Number'),ylabel('Track Error(red)/Error Predictor(blue)');
    if (isfield(opt,'dump') && opt.dump > 0) imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidNameW,datetoday,'Track Error Vs Track points differenceCorr_',num2str(correlation(2)),'.png']);end
    %% Filtered along time display data
    correlationFilttime=corr(filteredTrkptsErr',filteredTrkptsDiff');
    figure,plot(filteredTrkptsDiff),title(['Med filt along time Track points difference with Correlation .',num2str(correlationFilttime),'_Mean Error:',num2str(mean(meanerr))],'FontSize',tsize);
    if (isfield(opt,'dump') && opt.dump > 0)imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidNameW,datetoday,'Median filtered along time Track points difference.png']);end
    figure,plot( filteredTrkptsErr,'r'),title(['Median filt alng time Track Error with Correlation .',num2str(correlationFilttime),'_Mean Error:',num2str(mean(meanerr))],'FontSize',tsize);
    if (isfield(opt,'dump') && opt.dump > 0)imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,vidNameW,datetoday,'alng time Median filtered Track Error.png']);end
    [filteredTrkptsDiffN ] = normalizeData( filteredTrkptsDiff' );
    [filteredTrkptsErrN] = normalizeData( filteredTrkptsErr');
    figure,plot(filteredTrkptsErrN(1:5:end),'r')
    hold on
    plot(filteredTrkptsDiffN(1:5:end),'b'),xlabel('Frame Number'),ylabel('Track Error(red)/Error Predictor(blue)');
    if (isfield(opt,'dump') && opt.dump > 0)imwrite(frame2im(getframe(gcf)),[pathtoSaveResults,'Stem',vidNameW,datetoday,'alng time Track Error Vs Track points differenceCorr_',num2str(correlationFilttime),'.png']);end
    
    
    
    if(exist('showbasisdiff'))
        figure,  plot( totaldiffBasis,'g.-'),title('TotalDiffin Basis');
        totaldiffBasis1=totaldiffBasis(totaldiffBasis>0);
        totaldiffBasisNormal=normalizeData(totaldiffBasis1');
        for iter=1:numel(totaldiffBasis)
            
            
            if(totaldiffBasis(iter)==0)
                state(iter)=1;
                %                tline(7:20)=strcat(tline,'01 02 03 04');
                %                fprintf(fileid,'%s',tline);
            else
                
                state(iter)=0;
            end
            
        end
        n_filled=0;
        mask=[-1 1];
        changes=conv(state,mask,'same');
        startpoint=find(changes==-1);
        finalpoint = find(changes==1)+1;
        endbefore= finalpoint(end)-finalpoint(end-1);
        
        % startpoint=finalpoint-16;
        noofGaps=size(startpoint,2);
        noofmiss=zeros(1,noofGaps);
        totaldiffBasis2=totaldiffBasis;
        for i = 1:numel(totaldiffBasis)-endbefore
            if(n_filled<noofGaps-1)
                spoint = startpoint(n_filled+1);
                fpoint = finalpoint(n_filled+2);
                if(fpoint<numel(totaldiffBasis))
                    svalue = totaldiffBasis(spoint);
                    fvalue = totaldiffBasis(fpoint);
                    missing_points = linspace(svalue,fvalue,fpoint-spoint);
                    for ind=1: size(missing_points,2)
                        totaldiffBasis2(spoint+ind) = round(missing_points(1,ind));
                    end
                end
                n_filled = n_filled + 1;
            end
            
        end
        
        correlation4=corr(totaldiffBasis2',trackerr')
        [totaldiffBasisN] = normalizeData( totaldiffBasis2');
        figure,plot(trackerr1,'r')
        hold on
        plot(totaldiffBasis,'b'),title([TrackingMethod,'-Med filt Trck Err Vs IC Basis diff with Corr .',num2str(correlation4),'_Mean Error:',num2str(mean(meanerr)), 'ReInit Num:', num2str(mycount)],'FontSize',tsize);
        imwrite(frame2im(getframe(gcf)),[vidNameW,datetoday,'Med filt Trck Err Vs IC Basis diff with Corr_',num2str(correlation4),'.png']);
        
        
    end
    %% Stem plots for Error and predictor
    x=(1:30:numel(filteredTrkptsErrN));
    figure,stem(x,trackerr1(1:30:end),'^')
    hold on
    stem(x,trkptsdiff1(1:30:end),'o'),xlabel('Frame Number'),ylabel('Track Error(triangle)/Error Predictor(circle)');
    if (isfield(opt,'dump') && opt.dump > 0) imwrite(frame2im(getframe(gcf)),['Stem Track Error Vs Track points differenceCorr_',num2str(correlation1(2)),'.png']);end
    
    figure,stem(x,filteredTrkptsDiff1(1:30:end),'o'),xlabel('Frame Number'),ylabel('Track Error(triangle)/Error Predictor(circle)');
    hold on
    stem(x,filteredTrackErr1(1:30:end),'^')
    if (isfield(opt,'dump') && opt.dump > 0) imwrite(frame2im(getframe(gcf)),['Stem Filtered Track Error Vs Track points differenceCorr_',num2str(correlation1(2)),'.png']);end
    
    
    figure,stem(x,filteredTrkptsDiffN(1:30:end),'o')
    hold on
    stem(1:30:size(filteredTrkptsErrN,1),filteredTrkptsErrN(1:30:end),'^')
    ,xlabel('Frame Number'),ylabel('Track Error(triangle)/Error Predictor(circle)');
    if (isfield(opt,'dump') && opt.dump > 0)imwrite(frame2im(getframe(gcf)),['Stem alng time Track Error Vs Track points differenceCorr_',num2str(correlationFilttime),'.png']);end
% cd (programpath)
end
pathsGT=pathtoSaveResults;
save([pathsGT,'trackpts.mat'],'trackpts');
save([pathsGT,'error_per_image.mat'],'error_per_image');
save([pathsGT,'ErrorCenter.mat'],'ErrorCenter');
save([pathsGT,'trackerr.mat'],'trackerr');
save([pathsGT,'meanerr.mat'],'meanerr');

% sortedError=sort(ErrorCenter);


%% Denocio Trails
% imgforDeno=reshape(wimgs(:,1),[32 32]);
% firstCoef=CoefWinsCurrent(:,1);
% CoefImg=firstCoef;
% [dX,dY]=meshgrid(1:32,1:32);
% figure,surf(dX,dY,firstCoef);
%
%

end