function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%definimos la funcion h(x) para logistic regressions
h = sigmoid(X*theta);
%hacemos los logaritmos naturalesde la matriz h 1-h
loged_h=log(h);
loged_uno_h=log(1-h);
%Definimos el contenido del sumatorio par ala formual de coste
Jcontent = (-(y.')*loged_h-(1-y.')*loged_uno_h);
J =(1/m)*Jcontent;
%Definimos el contenido del sumatorio del gradient
gradContent = transpose((h-y))*X;
grad = (1/m)*gradContent;







% =============================================================

end
