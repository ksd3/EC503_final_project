function  k_best = knn_bank_find_k(Xtest,Ytest, Xtrain, Ytrain) 

N = size(Ytest,1);

%k values to evaluate 
k_val = 2:1:21;
accuracy = zeros(length(k_val),1);

for k = 1:length(k_val) 

% Calculate distances between test data points and training data points
distances = pdist2(Xtest, Xtrain); 

% Find the k nearest neighbors for each data point
[~, idx] = sort(distances, 2);

% Make predictions based on k nearest neighbors majoirty class
Ypred = mode(Ytrain( idx(:,1:k_val(k))   ), 2);
% calculate the error rate
accuracy = sum(Ytest~=Ypred)/N;
end

%find the index to the k value for lowest error rate
[best_accuracy,k_idx]=min(accuracy);
k_best=k_val(k_idx);




%scatter(k_val,accuracy)
plot(k_val,accuracy,'-o')

 ylabel('Error Rate')
 xlabel('k-value')
 title('Error rate for k-values ')



