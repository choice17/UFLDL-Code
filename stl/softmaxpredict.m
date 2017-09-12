function predict = softmaxpredict(input_features,theta)

X = input_features;
[n,m] = size(X);
W = reshape(theta,n,[]);
W = [W zeros(n,1)];
[~,predict] = max((X'*W)',[],1);

