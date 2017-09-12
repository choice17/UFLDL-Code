function displayCNNneuron(opttheta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses)
% display trained filters in an array
% displayCNNneuron(opttheta,imageDim,filterDim,numFilters,...
%                       poolDim,numClasses)

[Wc, ~, ~, ~] = cnnParamsToStack(opttheta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);
                    
                    displayData(reshape(Wc,[],numFilters)');