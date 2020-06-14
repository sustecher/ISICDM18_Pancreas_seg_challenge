addpath(genpath('C:\Users\hkuan\Desktop\matlab_additional')); 
clc;clear
datapath='G:\MSD_pancreas\data_raw\';
dirs=dir([datapath 'imagesTr']);
for k=1:1:281
    k

    data_seg=load_untouch_nii([datapath 'labelsTr\' dirs(k+2).name]);
    Label_3D=double(data_seg.img);TGS=double(Label_3D==2);PGS=double(Label_3D>0);
    %{
    data=load_untouch_nii([datapath 'imagesTr\' dirs(k+2).name]);
    X=data.img;
    X=permute(X,[3,2,1]);X=X(size(X,1):-1:1,size(X,2):-1:1,size(X,2):-1:1);
    spacing=data.hdr.dime.pixdim;
    
    
    PGS=permute(PGS,[3,2,1]);PGS=PGS(size(X,1):-1:1,size(X,2):-1:1,size(X,2):-1:1);
    TGS=permute(TGS,[3,2,1]);TGS=TGS(size(X,1):-1:1,size(X,2):-1:1,size(X,2):-1:1);
    
    Temp=double(X(TGS>0.5));
    %}
    %histogram(Temp);
    %TTT(k,1)=mean(Temp);
    %TTT(k,2)=std(Temp);
    %[TTT(k,1),TTT(k,2)]
    %[data.hdr.dime.scl_slope,data.hdr.dime.scl_inter]
    %TTT(k,1)=data.hdr.dime.scl_slope;
    %TTT(k,2)=data.hdr.dime.scl_inter;
    [MinA,MaxA,MinB,MaxB,MinC,MaxC] = Find_range(PGS,0,0,0);
    TTT(k,1)=MinA;
    TTT(k,2)=MinB;
    TTT(k,3)=MinC;
    TTT(k,4)=MaxA-MinA+1;
    TTT(k,5)=MaxB-MinB+1;
    TTT(k,6)=MaxC-MinC+1;
    TTT(k,:)
    %Image_size(k,1)=size(X,1);Image_size(k,2)=size(X,2);Image_size(k,3)=size(X,3);
    %Image_Resolution(k,1)=data.hdr.dime.pixdim(2);
    %Image_Resolution(k,2)=data.hdr.dime.pixdim(3);
    %Image_Resolution(k,3)=data.hdr.dime.pixdim(4);
    %Image_size(k,:)
    %Image_Resolution(k,:)
end

%%
%{
LR=mean(Temp)-3*std(Temp);
HR=mean(Temp)+3*std(Temp);
sli=45;
Img_=squeeze(X(sli,:,:));
yP_=squeeze(PGS(sli,:,:));

Img_T=Img_;Img_T(Img_T>HR)=HR;Img_T(Img_T<LR)=LR;
Img_T=Img_to_show(Img_T);
[Img_BW] = Img_GS_to_Img3(Img_T,generate_large_contour(yP_,3),1,2);
subplot(1,2,1),imshow(Img_T,[]);
subplot(1,2,2),imshow(Img_BW,[]);
%}

%% ISICDM data check
