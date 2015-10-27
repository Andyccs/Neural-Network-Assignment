function main_train_validate(spread, number_of_neurons)

rng('default');

%% Loading data
load('california_data');

%% Setting initial variables
train_set = P_train';
train_set_class = T_train';

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

%% Radial Basis Function
house_rbf(train_set, train_set_class, validation_set, validation_set_class, spread, number_of_neurons);

end