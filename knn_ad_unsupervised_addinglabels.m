% Read CSV file
df = readtable('^GSPC.csv');
df = df(2957:end, :);
df.Date = datetime(df.Date, 'InputFormat', 'yyyy-MM-dd');

% Create a plot
figure;
plot(df.Date, df.Close, 'b');
xlabel('Date');
ylabel('Closing values/USD');
title('Stock Data for Closing Values - S&P500');
grid on;

% Drop unnecessary columns
df(:, {'Open','High', 'Low','Volume', 'AdjClose'}) = [];
% Create a new variable named "ID"
df.ID = (0:height(df)-1)';

% Calculate pairwise distances using "ID" variable
k = 11;
distances = pdist2(table2array(df(:,2)), table2array(df(:,2)), 'euclidean');

% Sort distances and get indices of k+1 nearest neighbors
[sortedD, idx] = sort(distances, 2);

% Replace "Date" column with "ID" column
df.Date = df.ID;

% Analyze mean distances
distances_mean = mean(sortedD(:,2:k+1), 2);
%distances_mean_normalized = distances_mean / max(distances_mean);

% Set outlier threshold and find indices of outlier values
th_percentile = prctile(distances_mean,90); 
outlier_index = find(distances_mean > th_percentile);

% Assign scores to anomalies
anomaly_scores = zeros(size(df,1), 1);
anomaly_scores(outlier_index) = distances_mean(outlier_index);

if anomaly_scores(outlier_index) > 0

    anomaly_scores(anomaly_scores > 0) = 1;

end 

% Display outlier values
outlier_values = df(outlier_index, :);

figure;
plot(df.Date, df.Close, 'b');
hold on;
scatter(outlier_values.Date, outlier_values.Close, 'or');
xlabel('Date');
xlim([0 1075]);
xticks(linspace(0, 1075, 10));
xticklabels({'Jan 2019', 'Jul 2019', 'Jan 2020', 'Jul 2020', 'Jan 2021', 'Jul 2021', 'Jan 2021', 'Jan 2022','Jul 2022', 'Jan 2023'}); 
ylabel('Closing price/USD');
title('S&P500 Stock Prices');
legend('Data','Anomalies')
grid on;
hold off;