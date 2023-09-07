% Load data
data = readtable('bank-additional-full_normalised.csv');

X = table2array(data(:, 1:2));   % Features
Y = table2array(data(:, end));   % Labels

cv = cvpartition(size(data, 1), 'HoldOut', 0.3); % Need to do cvpartition without library
Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));
Xtest = X(test(cv), :);
Ytest = Y(test(cv));

% Set k value % need to do k fold validation 
k = knn_bank_find_k(Xtest,Ytest, Xtrain, Ytrain)   %k = 13;

% Calculate distances between test data points and training data points
distances = pdist2(Xtest, Xtrain);

% Find the k nearest neighbors for each data point
[~, idx] = sort(distances, 2);

% Make predictions based on k nearest neighbors
Ypred = mode(Ytrain(idx(:,1:k)), 2);


% Initial values 
tp=0; fp=0; tn=0; fn=0;

for i=1:length(Ypred)
    if Ypred(i)==1
        if Ytest(i)==1
            tp=tp+1;
        else
            fp=fp+1;
        end
    else
        if Ytest(i)==1
            fn=fn+1;
        else
            tn=tn+1;
        end
    end
end

tpr=tp/(tp+fn);
fpr=fp/(fp+tn);


%Calculate TPR and FPR for various threshold values
thresholds = 0:0.01:100;
tprs = zeros(size(thresholds));
fprs = zeros(size(thresholds));

for i = 1:length(thresholds)
   thresh = thresholds(i);
   tprs(i) = sum((Ypred >= thresh) & (Ytest == 1))/sum(Ytest == 1);
   fprs(i) = sum((Ypred >= thresh) & (Ytest == 0))/sum(Ytest == 0);
end

%Plot the ROC curve
figure
plot(fprs, tprs, 'LineWidth', 2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve')
axis square

hold on; 
plot([0 1], [0 1]); 
AUC = abs((trapz(fprs, tprs)));
legend('KNN', 'Random Classifier'); 
title('ROC Curve');
