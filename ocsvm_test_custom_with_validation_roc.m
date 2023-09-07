% Load the training, validation, and testing data
training_data = readtable('GSPC_training_labeled.csv');
validation_data = readtable('GSPC_validation_labeled.csv');
testing_data = readtable('GSPC_testing_labeled.csv');

% Extract ground truth labels
ytrain = training_data.Label;
yvalidation = validation_data.Label;
ytest = testing_data.Label;

% Extract all columns except for date
% Xtrain = table2array(training_data(:, 2:end));
% Xvalidation = table2array(validation_data(:, 2:end));
% Xtest = table2array(testing_data(:, 2:end));

% Extract the difference between opening and closing prices as the feature
Xtrain = training_data.Close - training_data.Open;
Xvalidation = validation_data.Close - validation_data.Open;
Xtest = testing_data.Close - testing_data.Open;
Xtrain = Xtrain(:);
Xvalidation = Xvalidation(:);
Xtest = Xtest(:);

% Train the one-class SVM on the training data and validate
nu_values = 0.001:0.001:0.1;
kernel_function = @(x, y) x * y'; % Linear Kernel
% kernel_function = @(x, y) exp(-norm(x - y)^2 / (2 * (1 / nu)^2)); % RBF kernel
best_nu = nu_values(1);
best_validation_score = -Inf;

for nu = nu_values
    [alpha, b] = one_class_svm(Xtrain, nu, kernel_function);
    
    score = zeros(size(Xvalidation, 1), 1);
    for i = 1:size(Xvalidation, 1)
        k = arrayfun(@(j) kernel_function(Xvalidation(i, :), Xtrain(j, :)), 1:size(Xtrain, 1));
        score(i) = alpha' * k';
    end
    score = score - b;
    
    validation_score = sum(score);

    disp(validation_score);
    
    if validation_score > best_validation_score
        best_validation_score = validation_score;
        best_nu = nu;
    end
end

% Train the one-class SVM on the training data using the best nu value
[alpha, b] = one_class_svm(Xtrain, best_nu, kernel_function);

% Evaluate the one-class SVM on the testing data
score = zeros(size(Xtest, 1), 1);
for i = 1:size(Xtest, 1)
    k = arrayfun(@(j) kernel_function(Xtest(i, :), Xtrain(j, :)), 1:size(Xtrain, 1));
    score(i) = alpha' * k';
end
score = score - b;

% Determine a threshold for detecting anomalies
pctile_score = prctile(score, 95);

% Detect anomalies in the testing data
anomaly_indices = find(score > pctile_score);
num_anomalies = length(anomaly_indices);
percent_detected = numel(anomaly_indices) / numel(testing_data.Date) * 100;

% Print the results
fprintf('Detected %d anomalies (%.2f%%)\n', numel(anomaly_indices), percent_detected);

% Plot the data with anomalies highlighted
figure;
plot(testing_data.Date, testing_data.Close);
hold on;
plot(testing_data.Date(anomaly_indices), testing_data.Close(anomaly_indices), 'ro');
title('Stock Data with Anomalies Highlighted');
xlabel('Date');
ylabel('Price');
legend('Data', 'Anomalies');

% Calculate TPR and FPR at various thresholds for the testing data
thresholds = linspace(min(score), max(score), 100);
TPR = zeros(length(thresholds), 1);
FPR = zeros(length(thresholds), 1);

for t = 1:length(thresholds)
    threshold = thresholds(t);
    predictions = score > threshold;
    
    TP = sum(predictions & ytest);
    FP = sum(predictions & ~ytest);
    FN = sum(~predictions & ytest);
    TN = sum(~predictions & ~ytest);
    
    TPR(t) = TP / (TP + FN);
    FPR(t) = FP / (FP + TN);
end

% Sort FPR and TPR in ascending order
[FPR_sorted, sorted_indices] = sort(FPR);
TPR_sorted = TPR(sorted_indices);

% Calculate AUC
AUC = trapz(FPR_sorted, TPR_sorted);

% Plot the ROC curve
figure;
plot(FPR, TPR);
hold on;
plot([0 1], [0 1]);
legend('OCSVM', 'Random Selection');
title(sprintf('ROC Curve (AUC = %.2f)', AUC));
xlabel('False Positive Rate');
ylabel('True Positive Rate');
