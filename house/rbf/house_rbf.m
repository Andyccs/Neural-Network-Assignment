function house_rbf(...
    train_set,...
    train_set_class,...
    validation_set,...
    validation_set_class,...
    number_of_cluster)

% variable
number_of_inputs = size(train_set, 1);

%% Preprocess

% Normalise train_set
[train_set, process_setting] = mapstd(train_set');
train_set = train_set';
train_set_mean = process_setting.xmean;
train_set_std = process_setting.xstd;

[train_set_class, process_setting] = mapstd(train_set_class');
train_set_class = train_set_class';
train_set_class_mean = process_setting.xmean;
train_set_class_std = process_setting.xstd;

% Normalize validation set
validation_set = trastd(validation_set', train_set_mean, train_set_std)';
validation_set_class = trastd(validation_set_class', train_set_class_mean, train_set_class_std)';

%% K mean clustering
[results, centroids] = kmeans(train_set, number_of_cluster, 'MaxIter', 1000);

%% Calculate covariance
correlation = cell(number_of_cluster, 1);

for i = 1 : number_of_cluster
    all_related_inputs = train_set(find(results == i), : );
    correlation{i} = inv(cov(all_related_inputs));
end

%% Calculate Gram Matrix
gram = calculate_gram(train_set, centroids, correlation);

%% Calculate weight
weight = inv(gram' * gram) * gram' * train_set_class;

%% Calculate Misclassification error using validation set
gram = calculate_gram(validation_set, centroids, correlation);
y = gram * weight;

%% Calculate misclassification errors
validation_set_class_original = poststd(validation_set_class, train_set_class_mean, train_set_class_std);
y_original = poststd(y, train_set_class_mean, train_set_class_std);
err = sqrt(mean((validation_set_class_original - y_original).^2));

fprintf('Misclassfication error: %f', err);

[validation_set_class_sorted, idx] = sort(validation_set_class_original, 1);
y_sorted = y_original(idx, :);

figure
scatter(1:size(validation_set_class_sorted), validation_set_class_sorted);
hold on
scatter(1:size(y_sorted), y_sorted, '+');

end