function [ Img ] = Enhance_contrast_ByTruncation(Img,LR,HR)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
%Img=double((Img_2D-min(Img_2D(:)))./(max(Img_2D(:))-min(Img_2D(:))));
%Img=Img_2D;

Img(Img<LR)=LR;
Img(Img>HR)=HR;
Img = (Img -LR)./(HR-LR);
end

