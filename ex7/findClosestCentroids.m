function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
% Number of examples
m = size(X,1);
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
%Vector para almacenar el valor de cada norma.
result = zeros(K,1);
%Recorremos cada ejemplo
for i =1:m
    %Dentro, recorremos cada centroide
    for j = 1:K
        ejemplo =X(i,:);
        centroide =centroids(j,:);
        %Hacemos la norma
        norma =(ejemplo-centroide).^2;
        result(j) = sum(norma);
    end
    %Obtenemos el minimo valor de las normas anteriores
    [value,index]=min(result);
    %Asociamos la posicion del minimo al vector de indices
    idx(i)=index;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

