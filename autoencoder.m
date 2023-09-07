% Set the parameters
X=readtable('bank-additional-full_normalised.csv'); X=table2array(X); Y=X(:,end); X=X(:,1:end-1);
input_layer_size = size(X, 2);   % Number of input dimensions
d=input_layer_size;
%Y=labels;
m=size(X,1);
hidden_layer_size = 10;          %Size of the hidden layer
lambda = 0.001;                  %Regularization parameter
epsilon = 0.1;                   %Initialize weights randomly from [-epsilon, epsilon]
num_iters = 1000;                %Number of iterations for optimization algorithm
alphaholder = 0.01:0.01:0.05;                    %Learning rate for optimization algorithm
size(alphaholder);
tprholder=zeros(length(alphaholder));
fprholder=zeros(length(alphaholder));
aucholder=zeros(1,length(alphaholder));

train_size = round(m * 0.6); %create test set
test_size = round(m * 0.2); %create training set
val_size = m - train_size - test_size;
idx = randperm(m);
X_train = X(idx(1:train_size), :); Y_train=Y(idx(1:train_size),:);
X_test = X(idx(train_size+1:train_size+test_size), :); Y_test=Y(idx(train_size+1:train_size+test_size),:);
X_val = X(idx(train_size+test_size+1:end), :); Y_val=Y(idx(train_size+test_size+1:end),:);

%Initialize the weights for the autoencoder randomly
W1 = rand(hidden_layer_size, input_layer_size) * 2 * epsilon - epsilon; W3=W1; 
W2 = rand(input_layer_size, hidden_layer_size) * 2 * epsilon - epsilon; W4=W2;
for ok=1:length(alphaholder)
    alpha=alphaholder(ok);
%Train the autoencoder using backpropagation
for i = 1:num_iters
    %Forward propagation
    z2 = X_train * W1'; %Mx62 x 62xhls=Mxhls
    a2 = logsig(z2);
    z3 = a2 * W2'; %reconstruct
    a3 = logsig(z3);
    
    %Calculate the reconstruction error
    error = a3 - X_train;
    
    % Backpropagation
    delta3 = error .* logsigGradient(z3);
    delta2 = (delta3 * W2) .* logsigGradient(z2);
    %size(delta2)
    %Add regularization terms to the gradients
    W1_grad = (delta2' * X_train)/m + lambda * W1;
    W2_grad = (delta3' * a2)/m + lambda * W2;
    
    %Update the weights
    W1 = W1 - alpha * W1_grad;
    W2 = W2 - alpha * W2_grad;
end

%Use the trained autoencoder to reconstruct the data
z2 = X_val * W1';
a2 = logsig(z2);
z3 = a2 * W2';
a3 = logsig(z3);

%Calculate the reconstruction error for each data point
reconstruction_errors = sum((a3 - X_val) .^ 2, 2);

%Evaluate the performance of the autoencoder for anomaly detection
threshold = prctile(reconstruction_errors, 90);   %Set the threshold to the 95th percentile
predictions = reconstruction_errors > threshold;  %Predict an anomaly if the reconstruction error is above the threshold
accuracy = sum(predictions == Y_val) / length(Y_val);     %Calculate the accuracy of the predictions
disp(['Accuracy: ', num2str(accuracy)]);
[~,~,~,auc]=perfcurve(Y_val,double(predictions),1);
aucholder(ok)=auc;

end
%disp(size(aucholder));

[~,maxauc]=max(aucholder); %get the index of the maximum auc
%disp(maxauc)
finalalpha=alphaholder(maxauc);
disp(finalalpha);
for i = 1:num_iters
    %Forward propagation
    z2 = X_test * W3'; %Mx62 x 62xhls=Mxhls
    a2 = logsig(z2);
    z3 = a2 * W4'; %reconstruct
    a3 = logsig(z3);
    
    %Calculate the reconstruction error
    error = a3 - X_test;
    %size(error)
    
    %Backpropagation
    delta3 = error .* logsigGradient(z3);
    %size(delta3)
    %size(W3)
    delta2 = (delta3 * W4) .* logsigGradient(z2);
    
    %Add regularization terms to the gradients
    W3_grad = (delta2' * X_test)/m + lambda * W3;
    W4_grad = (delta3' * a2)/m + lambda * W4;
    
    %Update the weights
    W3 = W3 - finalalpha * W3_grad;
    W4 = W4 - finalalpha * W4_grad;
end

%Use the autoencoder to reconstruct the data
z2 = X_test * W3';
a2 = logsig(z2);
z3 = a2 * W4';
a3 = logsig(z3);

reconstruction_errors = sum((a3 - X_test) .^ 2, 2);
threshold = prctile(reconstruction_errors, 90);   %Set the threshold to the 95th percentile
predictions = reconstruction_errors > threshold;  %Predict an anomaly if the reconstruction error is above the threshold
accuracy = sum(predictions == Y_test) / length(Y_test);     %Calculate the accuracy of the predictions
disp(['Accuracy of reconstruction on test set: ', num2str(accuracy)]);
[a,b,~,auc]=perfcurve(Y_test,double(predictions),1);
plot(a,b);


function g = logsigGradient(z)
%Calculate the derivative of the sigmoid function
g = logsig(z) .* (1 - logsig(z));
end
