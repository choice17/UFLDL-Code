function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);

activationsPooled = cnnPool(poolDim, activations);
% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer

activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
index = labels'==(1:numClasses)';
norm = Wd*activationsPooled+bd;
norm = exp(norm - max(norm));
denorm = sum(norm);
probs = norm./denorm;

%% sample
%{
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
probs = Wd * activationsPooled + repmat(bd, [1, numImages]);
probs = bsxfun(@minus, probs, max(probs, [], 1));
probs = exp(probs);
probs = bsxfun(@rdivide, probs, sum(probs));
%}
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

m = numImages;
lambda = 1e-4;
W = [Wc(:);Wd(:)];

%%% YOUR CODE HERE %%%
%cost = log(probs(index));
%cost = -sum(cost(:))/m ; + lambda/m/2*(W'*W);
cost = -(index(:)'*log(probs(:)))/m + lambda/2*(W'*W);

%% sample 
%{
groundTruth = full(sparse(labels, 1:numImages, 1));
cost = -(groundTruth(:)'*log(probs(:)))/m + (lambda/2.)*(sum(Wd(:).^2)+sum(Wc(:).^2));
%}
%%
% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
%{
numCnnLayer=1;
delta = cell(numCnnLayer+1,1);
%softmax delta
delta{numCnnLayer+1} = -(index - probs);
%conv delta
tempDelta = Wd'*delta{numCnnLayer+1};
tempDelta = reshape(tempDelta,outputDim,outputDim,numFilters,numImages);
% upsampling
for i = 1:numImages
    for j = 1:numFilters
        delta{numCnnLayer}(:,:,j,i) = (1./poolDim^2).*kron(squeeze(tempDelta(:,:,j,i)),ones(poolDim));
    end
end
clear tempDelta;
delta{numCnnLayer} = delta{numCnnLayer}.*(activations.*(1-activations));
%}
%% sample
delta = -(index - probs);
delta_pool = reshape(Wd'*delta, outputDim, outputDim, numFilters, numImages);
delta_conv = zeros(convDim,convDim,numFilters,numImages);
% upsampling the delta_pool to delta_conv
for i=1:numImages
    for j=1:numFilters
        delta_conv(:,:,j,i) = (1./poolDim^2) .* kron(squeeze(delta_pool(:,:,j,i)), ones(poolDim));
    end
end
delta_conv = activations .* (1-activations) .* delta_conv; 


%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
%{
Wc_grad = zeros(size(Wc));
Wd_grad = delta{numCnnLayer+1}*activationsPooled'./m + lambda.*Wd;
bd_grad = mean(delta{numCnnLayer+1},2)./m;
for i = 1:numFilters
    for j = 1:numImages
        tempfilter = squeeze(delta{numCnnLayer}(:,:,i,j));
        Wc_grad(:,:,i) = Wc_grad(:,:,i) + ...
            conv2(images(:,:,j),rot90(tempfilter,2),'valid');
    end
    bc_grad(i) = sum(reshape(delta{numCnnLayer}(:,:,i,:),[],1))/numImages;
end
Wc_grad = Wc_grad./numImages + lambda.*Wc;
clear timefilter;
%}
%% sample

numImages_inv=1/m;
Wd_grad = numImages_inv .* delta * activationsPooled' + lambda .* Wd;
bd_grad = numImages_inv .* sum(delta, 2);
Wc_grad = zeros(size(Wc));
for i=1:numFilters
    for j=1:numImages
        Wc_grad(:,:,i) = Wc_grad(:,:,i) + conv2(squeeze(images(:,:,j)),rot90(squeeze(delta_conv(:,:,i,j)),2),'valid');
    end
    Wc_grad(:,:,i) = numImages_inv .* Wc_grad(:,:,i) + lambda .* Wc(:,:,i);
    
    temp = delta_conv(:,:,i,:);
    bc_grad(i) = numImages_inv .* sum(temp(:));
end



%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
