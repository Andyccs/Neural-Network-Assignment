function main_train_validate_test(number_of_cluster)

rng('default');

%% Loading data
load('california_data');

%% Setting initial variables
train_set = P_train';
train_set_class = T_train';

test_set = P_test';
test_set_class = T_test';

%% Radial Basis Function
house_rbf(train_set, train_set_class, test_set, test_set_class, number_of_cluster);

end