%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%

[~,m] = size(x);

% cost function
L1 = sqrt((W*x).^2+params.epsilon);
L1cost = sum(L1(:))*params.lambda;
deltaL2 = W'*W*x-x;
L2cost = 0.5*(deltaL2(:)'*deltaL2(:));
cost = (L1cost + L2cost)/m;

% gradient calculation
deltaL1 = (W*x./(L1));
WgradL1 =  params.lambda.*deltaL1*x';
WgradL2 = W*deltaL2*x' + (W*x)*deltaL2';
Wgrad = (WgradL1+WgradL2) ./m;



% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);