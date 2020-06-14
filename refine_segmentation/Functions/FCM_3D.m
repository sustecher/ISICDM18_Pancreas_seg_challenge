function [dst,centers] = FCM_3D(src,cluster)

%dataset generation: intensity, distance varience
[rows,cols,highs] = size(src);
L = length(src(src>0));
X = zeros(L,1);
Y = zeros(L,1);
Z = zeros(L,1);
V = zeros(L,1);
t= 0;
for x=1:rows
    for y=1:cols
        for z = 1:highs
            if src(x,y,z)>0
               t=  t+1;
               X(t) = x;
               Y(t) = y;
               Z(t) = z;
               V(t) = src(x,y,z);
            end
        end
    end
end
% [X,Y,V] = find(src);
%
beta = 1;
V = beta*V;
fcmdata(:,1)=V;

%clusters using fuzzy c-means clustering.
options = [2 100 1e-5 false];
[centers,U] = fcm(fcmdata,cluster,options);

%Classify each data point into the cluster with the largest membership value.
maxU = max(U);
indexList = cell(1,cluster);
for n = 1:cluster
    index = find(U(n,:) == maxU);
    indexList{1,n}=index;
end
%image rebuilt
dst = zeros(rows,cols,highs);
for n = 1:cluster
    index = indexList{1,n};
    Xm = X(index,1);
    Ym = Y(index,1);
    Zm = Z(index,1);
    lengthIndex = length(index);
    for m = 1:lengthIndex
        x = Xm(m);
        y = Ym(m);
        z = Zm(m);
        dst(x,y,z)=n;
    end
end
end
