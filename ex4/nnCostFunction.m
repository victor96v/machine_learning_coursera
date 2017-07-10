function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Expand the 'y' output values into a matrix of single values 
%This is most easily done using an eye() matrix of size num_labels,
%with vectorized indexing by 'y'. 
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

bias = ones(m,1); %bias = a0
a1 = [bias X]; %input layer
z2 = a1*(Theta1.');
a2 = [bias sigmoid(z2)];%hidden layer
z3 = a2*(Theta2.');
a3 = sigmoid(z3);%output layer

log_h = log(a3);
one_log = log(1-a3);

J_factor1 = trace((-(y_matrix.'))*log_h);%Factor 1 del sumatorio sin reg
J_factor2 = trace((1-(y_matrix.'))*one_log);%Factor 2 del sumatorio sin reg
J_content = J_factor1-J_factor2;%Resta de factores
J = (1/m)*J_content;
%Now we are going to make the cost regularization
%We need to exclud the column of bias units for the regularization
pow2_Theta1 = (Theta1(:,2:end)).^2;
pow2_Theta2 = (Theta2(:,2:end)).^2;
%Now we make the sums
Reg_theta1 = sum(sum(pow2_Theta1));
Reg_theta2 = sum(sum(pow2_Theta2));
%Operate all
Reg_total = (lambda/(2*m))*(Reg_theta1+Reg_theta2);
%Add to the unregularizated cost
J = J + Reg_total;
%Backpropagation part
delta3 = a3-y_matrix;
%For delta2 we dont need the cost asociated to bias
delta2 = (delta3*(Theta2(:,2:end))).*sigmoidGradient(z2);
%Getting the big delts
bigDelta1 = (delta2.')*a1;
bigDelta2 = (delta3.')*a2;
%Obtaining the gradients
Theta1_grad = (1/m)*bigDelta1;
Theta2_grad = (1/m)*bigDelta2;
%Now, Let's make 0 the first column (dont need to add for first column)
Theta1(:,1) = 0;
Theta2(:,1) = 0;
%scaled for lambda/m
scaledTheta1 = (lambda/m)*Theta1;
scaledTheta2 = (lambda/m)*Theta2;
%Add to unregularizated grad
Theta1_grad = Theta1_grad + scaledTheta1;
Theta2_grad = Theta2_grad + scaledTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
