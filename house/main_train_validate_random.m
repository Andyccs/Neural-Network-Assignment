function main_train_validate_random(...
  n_layers_number_of_neurons,...
  n_layers_transfer_functions,...
  learning_rate,...
  iterations)

rng('default');

%% Load the housing data
load('california_data');
train_set = P_train;
train_set_class = T_train;

%% Split data
% 16.67% test_set, 16.67% validation_set, 66.67% train_set

indexes = randsample(1:length(train_set), 4128);
validation_set = train_set(:, indexes);
validation_set_class = train_set_class(:, indexes);

temp_indexes = zeros(1, length(train_set));
for i=1 : length(indexes)
    temp_indexes(1, indexes(1, i)) = 1;
end

train_set_indexes = find(temp_indexes == 0);
train_set = train_set(:, train_set_indexes);
train_set_class = train_set_class(:, train_set_indexes);

%% Train neural network and calculate misclassification erros
err = house(...
    train_set,...
    train_set_class,...
    validation_set,...
    validation_set_class,...
    n_layers_number_of_neurons,...
    n_layers_transfer_functions,...
    learning_rate,...
    iterations);

fprintf('misclassification error: %f', err);

end