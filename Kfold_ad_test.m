df = readtable('^GSPC.csv');
df = df(1:end, :);
df.Date = datetime(df.Date, 'InputFormat', 'yyyy-MM-dd');

% Drop unnecessary columns
df(:, {'High', 'Low', 'Close', 'Volume', 'AdjClose'}) = [];

df.Date = (0:height(df)-1)';

% Set range of k values to try
k_range = 1:30;

% Set number of folds for cross-validation
num_folds = 5;

% Initialize array storing avg number of outliers for each k value
avg_num_outliers = zeros(size(k_range));

for k = 1:length(k_range)
    k_idx = k_range(k);
    fold_sizes = repmat(floor(size(df, 1) / num_folds), 1, num_folds);
    fold_sizes(1:mod(size(df, 1), num_folds)) = fold_sizes(1:mod(size(df, 1), num_folds)) + 1;
    cv_indices = mat2cell(randperm(size(df, 1))', fold_sizes, 1);
    num_outliers = zeros(num_folds, 1);

    for fold_idx = 1:num_folds
        test_indices = cv_indices{fold_idx};
        train_indices = cat(1, cv_indices{1:fold_idx-1}, cv_indices{fold_idx+1:end});
        distances = pdist2(table2array(df(train_indices, :)), table2array(df(test_indices, :)), 'euclidean');
        [sortedD, idx] = sort(distances, 1);
        %mean for distance between each test point and its k neighbour 
        distances_mean = mean(sortedD(2:k_idx+1, :), 1)';
        outlier_index = find(distances_mean > 5);
        num_outliers(fold_idx) = length(outlier_index);
    end

    avg_num_outliers(k) = mean(num_outliers);

end

% Plot average number of outliers for each k value
figure; 
plot(k_range, avg_num_outliers, 'o-');
xlabel('Number of Neighbors (k)');
ylabel('Average number of outliers');
title('Cross-validated outlier detection performance for k-NN');
