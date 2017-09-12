function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %
  
  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
%%% YOUR CODE HERE %%%

%regularization parameter
lambda = 1e-4;

H = theta'*X;
J = H-y;
f = J*J'/2/m + sum(theta(:).^2)/2*lambda ;
g = X*J' + theta.*lambda;

%{
for i=1:m
   dif = theta'*X(:,i) - y(i);
   f = f + dif * dif;
   g(:) = g(:) + X(:,i) .* dif;
end
f = f*0.5;
f;
%}



