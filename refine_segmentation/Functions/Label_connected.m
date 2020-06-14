function [seg_CC] = Label_connected(seg)
%UNTITLED6 此处显示有关此函数的摘要
% label connected and save the largest one
    seg=seg>0.5;
    [L,num] = bwlabeln(seg);
    A=zeros(num,1);
    
    for i=1:num
    A(i,1)=size(find(L==i),1);
    end
    
    seg_CC=zeros(size(seg));
    seg_CC(L==find(A==max(A)))=1;
    seg_CC=double(seg_CC);
    
end

