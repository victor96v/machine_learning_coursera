function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y);
n = length (theta);% number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%definimos la funcion h(x) para logistic regressions
h = sigmoid(X*theta);
%hacemos los logaritmos naturalesde la matriz h 1-h
loged_h=log(h);
loged_uno_h=log(1-h);
%Definimos el contenido del sumatorio para la formula de coste
%Este es el primer sumatorio de la misma, a falta de la regularizacion
Jcontent = (-(y.')*loged_h-(1-y.')*loged_uno_h);
%Esta es la parte regularizadora
Jcontent_reg = sum( ( theta( 2:size(theta) ) ).^2 );
%Funcion de coste Reg
J =(1/m)*Jcontent+(lambda/(2*m))*Jcontent_reg;

%En el caso del gradient tenemos que diferenciar para
%el caso de theta(1) y el resto
%Definimos el contenido del sumatorio del gradient
gradContent = transpose(X)*(h-y);
%gradient para theta(1)
grad(1) = (1/m)*gradContent(1);
%Parte de gradient regularizada
grad_reg = (lambda/m)*theta(2:size(theta));
grad(2:size(theta)) = (1/m)*gradContent(2:size(theta)) + grad_reg;




% =============================================================

end
