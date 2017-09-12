function [ cost, grad, pred_prob,hAct] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)
%% elu paramter test
 a =1;

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for i=1:numHidden+1
    if i==1
        hAct{i}= data;
    else 
        hAct{i} = stack{i-1}.W*hAct{i-1}+stack{i-1}.b;
    end
        switch ei.activation_fun
            case 'logistic' 
                hAct{i} = sigmoid(hAct{i});
            case 'tanh'
                hAct{i} = tanh(hAct{i});
            case 'relu'
                hAct{i}(hAct{i}<0)=0;
            case 'elu'
                hAct{i}(hAct{i}<0)=a.*(exp(hAct{i}(hAct{i}<0) - 1));
        end

end
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  norm = exp(stack{numHidden+1}.W*hAct{numHidden+1}+stack{numHidden+1}.b);
  denorm = sum(norm);
  pred_prob=norm./denorm;
  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
% softmax cost function numHidden+1 i.e. output layer
index = labels'==(1:ei.output_dim)';
norm = exp(stack{numHidden+1}.W*hAct{numHidden+1}+stack{numHidden+1}.b);
denorm = sum(norm);
pred_prob=norm./denorm;

cost = -log(pred_prob(index));

%adding weight decay
W=0;
for i=1:numHidden+1
    W = W+sum(sum(stack{i}.W.^2));
end
m = length(data(1,:));

cost = sum(cost(:))/m + ei.lambda/2/m*W;
%% compute gradients using backpropagation
%%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

delta = cell(numHidden+1,1);   
%delta in output layer for softmax
for i = numHidden+1:-1:1
    if i==numHidden+1
       delta{i} = -(index-pred_prob);
       gradStack{i}.W  =  delta{i}*hAct{i}'./m+ ei.lambda.*stack{i}.W./m;
       gradStack{i}.b  =  mean(delta{i},2);
    else
       delta{i} =  stack{i+1}.W'*delta{i+1};
       switch ei.activation_fun
          case 'logistic'
                delta{i} = delta{i}.*gLogSig(hAct{i+1});
          case 'tanh'
                delta{i} = delta{i}.*hAct{i+1}.*(1-hAct{i+1});
          case 'relu'
                delta{i}(hAct{i+1}<=0) = 0;
          case 'elu'
                delta{i}(hAct{i+1}<=0) = delta{i}(hAct{i+1}<=0)+a;
       end
       
       gradStack{i}.W  =  delta{i}*hAct{i}'+ei.lambda.*stack{i}.W./m;
       gradStack{i}.b  =  mean(delta{i},2);
    end
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);

% save temp for theta to allow stop
save temp.mat theta
end



