
clear all;
close all;
clc;
%% Copyright (C) Aasim khurshid and Jacob Scharcaski
% akhurshid@inf.ufrgs.br
% CaseName='MaleDash';
% CaseName='FemaleDash';
if (isunix)
    symb='/';
else
    symb='\';
end

folder = fileparts(which('runtrackerDictionaries_linux.m')); 
% Add that folder plus all subfolders to the path.
addpath(genpath(folder));
%     DataFileDir is where to store results.
    %     pathLabels is if groundtruth of landmark is available
    %     vidNamesall is where videos/images are located. It should have
    %     folders inside if there are multiple videos
    DataFileDir=[folder,symb,'Data',symb];
    vidNamesall=dir([folder,symb,'Data',symb]);
DataFileDir1=DataFileDir;
pathtoSaveResults=[folder,symb,'Results',symb];
vidNamesall1=vidNamesall(3:end);
alphaa=0.8;  % bias between reconstructin and classification dictionary. Can change for better perfomrance.
patchsize=8; %is optimized. Also can be changed to test the performance difference.
% for alphaa=1:1.0
nmff=0; % if dont want to do noise reduction, 1 otherwise
savallvideos=1;%If wants to save error in file
if(exist('savallvideos','var') && savallvideos==1)
    ErrorFileID=fopen([pathtoSaveResults,'ResultsMMDLFT','.txt'],'a');
    
end
% batchsize=3;
% fprintf(ErrorFileID,[ ' ,Batchsize: ',num2str(batchsize)]);
% fprintf(ErrorFileID,' \n ');

for indvid=1:length(vidNamesall1)
% for indvid=4:5
% for indvid=7:7
    %     DataFileDir='/media/aasim/E1/Aasim/Data sets/YawDD dataset/Test Videos_Tracking2016/Images/';
    
    vidName=vidNamesall1(indvid).name;
    pathLabels= [folder,symb,'Data',symb,vidName,symb];
    pathsGT=[pathtoSaveResults,vidName,symb];
    
    mkdir(pathsGT);
    dataFolder1=[DataFileDir,vidName,'/'];
    imagefiles =dir([dataFolder1,'*.png']);
    %     flabel=fopen([pathLabels1,'results_',vidName(1:end-4),'.txt'],'w');
    groundTruth=0;
    ReadData;
    % %     How to initialize the landmarks :(
    % % Using face_detectVJ and then CLM can be used to fix the landmarks
    
    alphaa=0.9;
    ErrorFileID=fopen(['MMDLFT_errorVideo','.txt'],'a');
    fprintf(ErrorFileID,vidName);
    fprintf(ErrorFileID,' \n ');
    options.assignweights=0;
    vidNameW=vidName;
    opt.drivervid=1; %if YawDD dataset is used.
    %         opt.drivervid=0; %if some other dataset is used.
    patch=1; %If want to do patch processing
    patch=0; %If dont want to do patch processing.
    patchsize=8; %optimize patch size for YawDD dataset, can change and test.
    batchsize=3; %Test for differnet batchsizes which works better for type of videos.
    DataFileDir=pathtoSaveResults;
    [dispstr,dispstr1]=runtrackerDictionaries_linux(vidNameW,alphaa,batchsize,patchsize,data,truepts,param0,first,my_mat_x,ErrorFileID,opt,pathsGT);
    close all;
end
fclose(ErrorFileID);
fclose all;
