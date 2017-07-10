function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%All posible values of sigma, c
values = [0.01 0.03 0.1 0.3 1 3 10 30];
error = magic(length(values));
for i = 1:length(values)
    for j = 1:length(values)
    C = values(i);
    sigma = values(j);
    %for each pair C,sigma, we wet the model
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    %predict the value
    predictions = svmPredict(model, Xval);
    %get the error for that pair
    error(i,j) = mean(double(predictions ~= yval));
    end
end
%k = find(X) returns a vector containing the linear indices 
%of each nonzero element in array X.

%[row,col] = find(___) returns the row and column subscripts of each 
%nonzero element in array X using any of the input arguments in previous syntaxes.

%Así encontramos la posicón del mínimo máximo de una matriz (error)
[C_opt_index,sigma_opt_index]=find(error==min(min((error))));
C = values(C_opt_index);
sigma = values(sigma_opt_index);






% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
