function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
%%% YOUR CODE HERE %%%
%linear
for i=1:m
    h = 1/(1+exp(-theta'*X(:,i)));
    f = f - y(i)*log(h) - (1-y(i))*log(1-h);
    g(:) = g(:) + X(:,i).*(h - y(i));
end
%vectorize
lambda = 1e-4;
H = sigmoid(theta'*X);
f = (-y*log(H)' - (1-y)*log(1-H)')/m + lambda/2*(theta'*theta);
g = X*(H-y)'./m;


