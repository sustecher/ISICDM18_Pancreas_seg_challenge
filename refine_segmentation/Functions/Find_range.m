function [MinA,MaxA,MinB,MaxB,MinC,MaxC] = Find_range(GS,marginX,marginY,marginZ)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
    %% 3D mode
        for i=1:1:size(GS,1)
            A(i)=max(max(GS(i,:,:)));
        end
            IDX_A=find(A~=0);
            MinA=min(IDX_A)-marginX;MaxA=max(IDX_A)+marginX;
            
        for j=1:1:size(GS,2)
            B(j)=max(max(GS(:,j,:)));
        end
            IDX_B=find(B~=0);
            MinB=min(IDX_B)-marginY;MaxB=max(IDX_B)+marginY;

        for k=1:1:size(GS,3)
            C(k)=max(max(GS(:,:,k)));
        end
            IDX_C=find(C~=0);
            MinC=min(IDX_C)-marginZ;MaxC=max(IDX_C)+marginZ;
            
            MinA=max(1,MinA);
            MinB=max(1,MinB);
            MinC=max(1,MinC);
            
            MaxA=min(size(GS,1),MaxA);
            MaxB=min(size(GS,2),MaxB);
            MaxC=min(size(GS,3),MaxC);
            
end