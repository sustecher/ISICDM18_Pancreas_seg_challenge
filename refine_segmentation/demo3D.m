clc;clear;
addpath(genpath('.\Functions\'))
% please downlod "Tools for NIfTI and ANALYZE" image throungh https://github.com/mcnablab/NIFTI_toolbox
clear all;
close all;
dowm_sampling_factor=1;
%%
idx='06'
data=load_untouch_nii(['./' idx '_SEGFine.nii.gz']);SEG_Fine_Ori=double(data.img);
data=load_untouch_nii(['./' idx '_Image.nii.gz']);Img_Ori=double(data.img);
%%
tic;
Size_Ori=size(Img_Ori);
spacing=data.hdr.dime.pixdim(2:4);

numrows=round(Size_Ori(1)*spacing(2)*dowm_sampling_factor);
numcols=numrows;
numplanes=round(Size_Ori(3)*spacing(3)*dowm_sampling_factor);

SEG_Fine = imresize3(SEG_Fine_Ori,[numrows numcols numplanes]);SEG_Fine(SEG_Fine >0.5)=1;SEG_Fine(SEG_Fine<0.5)=0;
Img= imresize3(Img_Ori,[numrows numcols numplanes]);
Pixels_SEGFine=Img(SEG_Fine==1);
Img=Enhance_contrast_ByTruncation(Img,prctile(Pixels_SEGFine,1),prctile(Pixels_SEGFine,99));
%%
[MinA,MaxA,MinB,MaxB,MinC,MaxC] = Find_range(SEG_Fine,10,10,10);
MaxC=min(MaxC,size(SEG_Fine,3));
SEG_Fine_crop = SEG_Fine(MinA:MaxA,MinB:MaxB,MinC:MaxC);
Img_crop = Img(MinA:MaxA,MinB:MaxB,MinC:MaxC);
SEG_PP = zeros(size(SEG_Fine));

[Img_crop_FCM,centers] = FCM_3D(Img_crop,3);
[~,sortFCM] = sort(centers);
Mask_out=(Img_crop_FCM~=sortFCM(3)).*(Img_crop_FCM>0);
se = strel('sphere',1);
Mask_out= imclose(Mask_out,se);
Mask_out(SEG_Fine_crop==1)=1;
Mask_out= imdilate(Mask_out,se);
Img_crop_Mask=Mask_out.*Img_crop;
%%
sigma=1;     % scale parameter in Gaussian kernel
G=fspecial3('gaussian',[5 5 5],[sigma,sigma,sigma]);
Img_smooth=convn(Img_crop_Mask,G,'same');  % smooth image by Gaussiin convolution
[Ix,Iy,Iz]=gradient(Img_smooth);
f=Ix.^2+Iy.^2+Iz.^2;
g=1./(1+f);  % edge indicator function.
g=histeq(g);

G=fspecial3('gaussian',[5 5 5],[sigma,sigma,sigma]);
g=convn(g,G,'same');   
g=Enhance_contrast_ByTruncation(g,prctile(g(:),1),prctile(g(:),99));
toc;
%% parameter setting
tic
timestep=2;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=1;
Max_iter_outer=20;
lambda=1.5; % coefficient of the weighted length term L(phi)
alfa=-1.5;% 1.5;  % coefficient of the weighted area term A(phi) %% -≈Ú’Õ  + ’Àı
epsilon=1.5; % papramater that specifies the width of the DiracDelta function

% initialize LSF as binary step function
c0=2;
initialLSF=c0*ones(size(Img_crop));
initialLSF(SEG_Fine_crop==1)=-c0;
phi=initialLSF;
potentialFunction = 'double-well'; %single-well  % use double-well potential in Eq. (16), which is good for both edge and region based models
% start level set evolution
Differ_iteration=[];
Dice_iteration=[];
for n=1:Max_iter_outer
    phi_before=phi;
    phi = drlse_edge_3D(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
end
%%
for n=1:Max_iter_outer
    phi_before=phi;
    phi = drlse_edge_3D(phi, g, lambda, mu, 0, epsilon, timestep, iter_inner, potentialFunction);
 end
toc

SEG_PP(MinA:MaxA,MinB:MaxB,MinC:MaxC)=double(phi<0);
g_whole=zeros(size(SEG_PP));
g_whole(MinA:MaxA,MinB:MaxB,MinC:MaxC)=g;
g_whole_Ori=imresize3(g_whole,Size_Ori);
SEG_PP_Ori=imresize3(SEG_PP,Size_Ori);SEG_PP_Ori(SEG_PP_Ori>0.5)=1;SEG_PP_Ori(SEG_PP_Ori<=0.5)=0;
%%
data.img=SEG_PP_Ori;
save_untouch_nii(data,['./'  idx '_SEGRefine.nii.gz'])

%% Quantative
data=load_untouch_nii(['./' idx '_GroundTruth.nii.gz']);GS_Ori=double(data.img);GS_Ori(GS_Ori>0.5)=1;
GS = imresize3(GS_Ori,[numrows numcols numplanes]);GS(GS >0.5)=1;GS(GS<0.5)=0;

tmp=Dice_Ratio(SEG_Fine_Ori,GS_Ori);
tmp2=Dice_Ratio(SEG_PP_Ori,GS_Ori); 
[tmp,tmp2]
%%
