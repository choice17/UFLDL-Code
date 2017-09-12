function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);
  ly = length(unique(y));
  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  theta(:,num_classes)=0;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%parameter for regularization
lambda = 1e-4;
%calc probability prob
index = y==(1:ly)';
norm = exp(theta'*X);
denorm = sum(norm);
prob = norm./denorm;

%full/sparse function groundTruth = full(sparse(y,1:m,1));
%calc cost function objective function
%adding regularization term
f = -log(prob(index)');
f = sum(f(:))'/m ;
f = f + 0.5*lambda*sum(theta(:).^2);

%%%
g = -X*(index - prob)'./m + lambda.*theta ;
g(:,num_classes)=[];
g = g(:); % make gradient a vector for minFunc
