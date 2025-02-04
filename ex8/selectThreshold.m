function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % validation set = pval, ground truth = yval
    
    % get a binary vector of 0's and 1's of the outlier predictions
    cvPredictions = (pval < epsilon);
    % we need prec, recall in order to get F1
    % For prec and rec, we ned true positives, false negatives and
    % false positives
    % You can then, for example, computethe number of false positives using: 
    fp = sum((cvPredictions == 1) & (yval == 0));
    fn = sum((cvPredictions == 0) & (yval == 1));
    tp = sum((cvPredictions == 1) & (yval == 1));
    %now we get prec and recall
    prec = (tp)/(tp+fp);
    recall = (tp)/(tp+fn);
    %Finally, let's get F1 score
    F1 = (2*prec*recall)/(prec+recall);
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions













    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
