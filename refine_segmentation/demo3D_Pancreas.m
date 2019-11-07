clc;clear;
clear all;
close all;
genPPT=0;
genBYU=1;
DeBUG=1;
Dice_PP=zeros(16,20);
for k=6
    p=k;
    filename=[num2str(p) '.mat']
    %filename=sprintf('%04d.mat',p)

    load(['H:\ISICDM_Mat\ISICDM_test_mat\GS\y_' filename]);GS=double(y);
    load(['H:\ISICDM_Mat\ISICDM_test_mat\Img\X_' filename]); Img=double(X);
    load(['H:\ISICDM_Mat\ISICDM_test_mat\pred\' num2str(p+20) '.mat']); SEG_2D=double(SEG);
    %load(['H:\ISICDM_Mat\ISICDM_test_mat\pred-X\' num2str(p+20) '.mat']); SEG_2DX=double(SEG_X);
    %load(['H:\ISICDM_Mat\ISICDM_test_mat\pred-Y\' num2str(p+20) '.mat']); SEG_2DY=double(SEG_Y);
    %load(['H:\ISICDM_Mat\ISICDM_test_mat\pred-Z\' num2str(p+20) '.mat']); SEG_2DZ=double(SEG_Z);

    %SEG_2D=zeros(size(Img));
    %SEG_2D(SEG_2DZ+SEG_2DY+SEG_2DZ>1.5)=1;
    %SEG_2D=Label_connected(SEG_2D);
    
    load(['H:\ISICDM_Mat\ISICDM_test_mat\Pred_3D\y_pred_' filename]); SEG_3D=double(pred3D);
    GS(GS==255)=1;
    SEG_3D(SEG_3D==255)=1;
    
    %SEG_PP=LevelSet_PP3D(SEG_2D,Img);
    %Dice_PP(k,1)=Dice_Ratio(SEG_PP>0.5,GS);
    %[k,Dice_PP(k,1)]
    LC_SEG_2D=Label_connected(SEG_2D>0.5);

    if DeBUG==1
        [MinA,MaxA,MinB,MaxB,MinC,MaxC] = Find_range(LC_SEG_2D,5);
        MaxC=min(MaxC,size(SEG_2D,3));
        SEG_2D_crop = LC_SEG_2D(MinA:MaxA,MinB:MaxB,MinC:MaxC);
        Img_crop = Img(MinA:MaxA,MinB:MaxB,MinC:MaxC);
        GS_crop = GS(MinA:MaxA,MinB:MaxB,MinC:MaxC);
        SEG_PP = zeros(size(SEG_2D));

        A=255;
        Img_crop=A*normalize01(Img_crop);
        %[x,y,z] = ndgrid(-3:3);   
        %se = strel(sqrt(x.^2 + y.^2 + z.^2) <=3);
        %SEG_2D_initial=imdilate(SEG_2D_crop,se);
        SEG_2D_initial=SEG_2D_crop;
        %% parameter setting
        timestep=1;  % time step
        mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
        iter_inner=1;
        iter_outer=12;
        lambda=1.5; % coefficient of the weighted length term L(phi)
        alfa=1.5;  % coefficient of the weighted area term A(phi)
        epsilon=1.5; % papramater that specifies the width of the DiracDelta function

        sigma=1.5;     % scale parameter in Gaussian kernel
        %G=fspecial('gaussian',[15 15 15],sigma);
        %Img_smooth=convn(Img_crop,G,'same');  % smooth image by Gaussiin convolution
        %Img_smooth=smooth3(Img_crop,'gaussian',3,sigma);
        
        Img_smooth=Img_crop;
        [Ix,Iy,Iz]=gradient(Img_smooth);
        f=Ix.^2+Iy.^2+Iz.^2;
        g=1./(1+f);  % edge indicator function.

        % initialize LSF as binary step function
        c0=-2;
        initialLSF=c0*ones(size(Img_crop)) - 2*c0*SEG_2D_initial;
        % generate the initial region R0 as a rectangle

        phi=initialLSF;

        %figure(1);
        %mesh(-phi);   % for a better view, the LSF is displayed upside down
        %hold on;  contour(phi, [0,0], 'r','LineWidth',2);
        %title('Initial level set function');
        %view([-80 35]);

        potential=2;  
        if potential ==1
            potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model 
        elseif potential == 2
            potentialFunction = 'double-well';  % use double-well potential in Eq. (16), which is good for both edge and region based models
        else
            potentialFunction = 'double-well';  % default choice of potential function
        end

        % start level set evolution
        for n=1:iter_outer
            phi = drlse3D_edge(initialLSF,phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
            D(n)=Dice_Ratio(Label_connected(phi>0),GS_crop);
            [n,D(n)]
        end
        
        %c0=2;
        %phi=c0*ones(size(Img_crop)) - 2*c0*phi;
        %for n=1:iter_outer
        %    phi = drlse3D_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
        %    D(n+iter_outer)=Dice_Ratio(Label_connected(phi<=0),GS_crop);
        %    [n+iter_outer,D(n+iter_outer)]
        %end

        %Dice_PP(k,:)=D;
       
    else
    end
    
end
%%
SEG_PP(MinA:MaxA,MinB:MaxB,MinC:MaxC)=Label_connected(phi>0);
for i=1:1:size(SEG_PP,3)
    if max(max(SEG_PP(:,:,i)))>0.5
        SEG_=SEG_PP(:,:,i);
        [L,num] = bwlabeln(SEG_);
        A=zeros(num,1);
        Zeros_SEG=zeros(size(SEG_));
        for j=1:num
            A(j,1)=size(find(L==j),1);
            if A(j,1)>10
                Zeros_SEG(L==j)=1;
            else
            end
        end   
        %Zeros_SEG=zeros(size(SEG_));
        SEG_PP(:,:,i)=Zeros_SEG;
        
    else
    end
    
end

Dice_Ratio(LC_SEG_2D,GS)
Dice_Ratio(SEG_PP,GS)
%%
if genBYU==1
    [Face,X_new] = Fun_byuGenerate(GS);
    savebyu(Face,X_new,['./BYU/' 'GS.byu'])
    [Face,X_new] = Fun_byuGenerate(LC_SEG_2D);
    savebyu(Face,X_new,['./BYU/' 'SEG_2D.byu'])
    [Face,X_new] = Fun_byuGenerate(SEG_PP>0.5);
    savebyu(Face,X_new,['./BYU/' 'SEG_2D_PP.byu'])
else
end

%% Gen PPT
%T=double(phi)>0;
%SEG_2D(MinA:MaxA,MinB:MaxB,MinC:MaxC)=T;
%[Face,X_new] = Fun_byuGenerate(SEG_2D);
%savebyu(Face,X_new,['SEG_2D_PP.byu']);
%

for i=1:1:size(Img,3)
    MaxV_Slice(i)=max(max(GS(:,:,i))); 
end
IDX=find(MaxV_Slice==1);

if genPPT==1
    Bsize=2;
    Img = Img_to_show(Img);
    isOpen  = exportToPPTX();

    if ~isempty(isOpen)
        % If PowerPoint already started, then close first and then open a new one
        exportToPPTX('close');
    end

    exportToPPTX('new','Dimensions',[12 6], ...
        'Title','Seg_results', ...
        'Author','Yue', ...
        'Subject','Automatically generated PPTX file', ...
        'Comments','This file has been automatically generated by exportToPPTX');

    % Additionally background color for all slides can be set as follows:
    % exportToPPTX('new','BackgroundColor',[0.5 0.5 0.5]);
    fileStats   = exportToPPTX('query');

    if ~isempty(fileStats)
        fprintf('Presentation size: %f x %f\n',fileStats.dimensions);
        fprintf('Number of slides: %d\n',fileStats.numSlides);
    end

    newFile = exportToPPTX('saveandclose','Seg_results');
    exportToPPTX('open','Seg_results');

    for slice_num=IDX
        slideId  = exportToPPTX('addslide');
        Img_= squeeze(Img(:,:,slice_num));
        GS_= squeeze(GS(:,:,slice_num));
        Img_3 = Img_GS_to_Img3(Img_,GS_,1);
        SEG_2D = Label_connected(SEG_2D>0.5);
        SEG_3D_= squeeze(SEG_3D(:,:,slice_num));
        SEG_2D_= squeeze(SEG_2D(:,:,slice_num));
        SEG_PP_= squeeze(SEG_PP(:,:,slice_num));
        SEG_PP_M= SEG_PP_-SEG_2D_;
        

        SEG_3D_3=DSC_imshow(GS_,SEG_3D_>0.5);  
        SEG_2D_3=DSC_imshow(GS_,SEG_2D_>0.5);
        SEG_PP_3=DSC_imshow(GS_,SEG_PP_>0.5);
        

    %% 

        exportToPPTX('addpicture',Img_,'Position',[0 0 Bsize Bsize])
        exportToPPTX('addpicture',GS_,'Position',[Bsize 0 Bsize Bsize])
        exportToPPTX('addpicture',SEG_3D_,'Position',[2*Bsize 0 Bsize Bsize])
        exportToPPTX('addpicture',SEG_2D_,'Position',[3*Bsize 0 Bsize Bsize])
        exportToPPTX('addpicture',SEG_PP_,'Position',[4*Bsize 0 Bsize Bsize])
        exportToPPTX('addpicture',SEG_PP_M,'Position',[5*Bsize 0 Bsize Bsize])
        
        exportToPPTX('addpicture',Img_3 ,'Position',[Bsize Bsize Bsize Bsize])
        exportToPPTX('addpicture',SEG_3D_3,'Position',[2*Bsize Bsize Bsize Bsize])
        exportToPPTX('addpicture',SEG_2D_3,'Position',[3*Bsize Bsize Bsize Bsize])
        exportToPPTX('addpicture',SEG_PP_3,'Position',[4*Bsize Bsize Bsize Bsize])


        for i=1:1:5
            for j=1:1:2
                exportToPPTX('addtext','','Position',[(i-1)*Bsize (j-1)*Bsize Bsize Bsize],'LineWidth',2,'EdgeColor',[1,1,1]);
            end
        end
    end
    exportToPPTX('saveandclose');
    
else
end

%% Level-set PP

%%