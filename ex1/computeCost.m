function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%   X = array de datos de entrada
%   y = array de datos de salida
%   theta = array de parámetros

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% Contenido del sumatorio
formula_sum =(X*theta-y).^2;
% Aplcamos los factores y realizamos el sumatorio
J = (1/(2*m))*(sum(formula_sum));

% =========================================================================
end
