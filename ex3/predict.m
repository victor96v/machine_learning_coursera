function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%Primero obtenemos la funcion h(x) para cada ejemplo, utilizando los nuevos
%parametros obtenidos del ejercicio onevsall. Por tanto, necesitamso hacer
%la sigmoide de X,all_theta.
%The expression C = [A B] horizontally concatenates matrices A and B.
%The expression C = [A; B] vertically concatenates them.
%Añadimos el vector columna de x0 (a0)
bias = ones(size(X, 1),1);
X = [bias X];
%Operamos para ir de la primera a la segunda capa
a1 = sigmoid(X*(Theta1.'));
%Añadimos la columna a0 al principio
a1 = [bias a1];
%Operamos para ir de la segunda a la 3era capa
a2 = sigmoid(a1*(Theta2.'));
%Obtenemos los valores y los indices de los máximos para cada ejemplo 
%help max for more info
[values,indexes] = max(a2, [], 2);
p= indexes;








% =========================================================================


end
