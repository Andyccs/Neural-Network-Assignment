rng('default');

clear; clc; close all;
%% Loading data
load('california_data');

%% Setting initial variables
train_set = P_train';
train_set_class = T_train';

test_set = P_test';
test_set_class = T_test';

number_of_cluster = 5;
number_of_inputs = size(train_set, 1);

%% K mean clustering
[results, centroids] = kmeans(train_set, number_of_cluster);

%% Calculate correlation
correlation = cell(number_of_cluster, 1);

for i = 1 : number_of_cluster
    all_related_inputs = train_set(find(results == i), : );
    distance = all_related_inputs - centroids(i, 1);
    correlation{i} = (1 / (size(all_related_inputs, 1) - 1)) * (distance' * distance);
end

%% Inverse correlation
for i = 1 : number_of_cluster
    correlation{i} = inv(correlation{i});
end

gram = zeros(number_of_inputs, number_of_cluster);

for i = 1 : number_of_cluster
    for j = 1 : number_of_inputs
        distance = train_set(j,:) - centroids(i, :);
        gram(j, i) = exp( -0.5 * (distance * correlation{i} * distance') );
    end
end

weight = inv(gram' * gram) * gram' * train_set_class;
weight

