function [ Img ] = Enhance_contrast_ByTruncation(Img,LR,HR)
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%Img=double((Img_2D-min(Img_2D(:)))./(max(Img_2D(:))-min(Img_2D(:))));
%Img=Img_2D;

Img(Img<LR)=LR;
Img(Img>HR)=HR;
Img = (Img -LR)./(HR-LR);
end

