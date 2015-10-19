rng('default');

clear; clc; close all;
%% Loading data
load('california_data');

%% Setting initial variables
train_set = P_train';
train_set_class = T_train';

test_set = P_test';
test_set_class = T_test';

number_of_cluster = 15;

%% Random split data
indexes = randsample(1:length(train_set), 4128);
validation_set = train_set(indexes, :);
validation_set_class = train_set_class(indexes, :);

temp_indexes = zeros(1, length(train_set));
for i=1 : length(indexes)
    temp_indexes(1, indexes(1, i)) = 1;
end

train_set_indexes = find(temp_indexes == 0);
train_set = train_set(train_set_indexes, :);
train_set_class = train_set_class(train_set_indexes, :);

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

% Normalize test set
test_set = trastd(test_set', train_set_mean, train_set_std)';
test_set_class = trastd(test_set_class', train_set_class_mean, train_set_class_std)';

%% K mean clustering
[results, centroids] = kmeans(train_set, number_of_cluster, 'MaxIter', 1000, 'Display', 'iter');

%% Calculate covariance
correlation = cell(number_of_cluster, 1);

for i = 1 : number_of_cluster
    all_related_inputs = train_set(find(results == i), : );
    correlation{i} = inv(cov(all_related_inputs));
end

%% Calculate Gram Matrix
gram = zeros(number_of_inputs, number_of_cluster);

for i = 1 : number_of_cluster
    for j = 1 : number_of_inputs
        distance = train_set(j,:) - centroids(i, :);
        gram(j, i) = exp( -0.5 .* (distance * correlation{i} * distance') );
    end
end

%% Calculate weight
weight = inv(gram' * gram) * gram' * train_set_class;

%% Calculate output
y = gram * weight;

%% Calculate errors
train_set_class_original = poststd(train_set_class, train_set_class_mean, train_set_class_std);
y_original = poststd(y, train_set_class_mean, train_set_class_std);
sqrt(mean((train_set_class_original - y_original).^2))
