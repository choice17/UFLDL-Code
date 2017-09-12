function [data_train,label_train] = loadData(X,y,ratio)
N = length(y);
R = randperm(N);
Rratio = R(1:ceil(N*ratio));
label_train = y(Rratio);
data_train = X(:,Rratio);
