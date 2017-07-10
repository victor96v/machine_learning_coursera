function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%sum_theta1=0;sum_theta2=0;
for iter = 1:num_iters
%obtenemos h(x)
hyp = X*theta;
%realizamos h(x)-y
error_vector = hyp-y;
%obtenemos (h(x)-y)*xj
sumat = transpose(X)*error_vector;
%aplicamos factores al sumatorio
change = (alpha/m)*sumat;
%actualizamos los valores
theta = theta - change;
% Save the cost J in every iteration    
J_history(iter) = computeCost(X, y, theta);

end
 
end
%% Codigo anterior
%     for n = 1:m
%         %Valor del sumatorio para i = n
%         hyp = theta(1)+theta(2)*X(n,2);
%         form_theta1 = (hyp-y(n))*X(n,1);
%         form_theta2 = (hyp-y(n))*X(n,2);
%        %Actualización del sumatorio para i = n
%        sum_theta1 = sum_theta1 + form_theta1;
%        sum_theta2 = sum_theta2 + form_theta2;
%     end
%     %Actualización de theta
%     theta(1) = theta(1) - (alpha/m)*sum_theta1;
%     theta(2) = theta(2) - (alpha/m)*sum_theta2;
