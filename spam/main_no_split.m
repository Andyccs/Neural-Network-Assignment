clear; clc; close all;
rng('default');

%% load the spam data
load('spam_data.mat');

%% Preprocessing

% preprocess training set by normalization to 0 mean and 1 std
[train_set, process_setting] = mapstd(P_train);
train_set_mean = process_setting.xmean;
train_set_std = process_setting.xstd;
train_set_class = T_train;

% transform test set by normalization to:
% 1. mean of training set
% 2. std of training set
test_set = trastd(P_test, train_set_mean, train_set_std);
test_set_class = T_test;

% test_data is a struct to hold test_set and test_set_class
test_data.P = test_set;
test_data.T = test_set_class;

% training set variables: train_set, train_set_class
% testing set variables: test_set, test_set_class, test_data
% validation set variables: validation_set, validation_set_class, validation_data

% uncomment this line if you want to visualize all 57 x 57 attributes
% plotData(test_set, test_set_class);

%% Setting up variables for feed forward networks
minmax_train_set = minmax(train_set);

number_of_hidden_neuron = 10;   % Experiment variable 2
n_layers_number_of_neurons = ... % Experiment variable 3
    [number_of_hidden_neuron, 10, 1];

% transfer function: activation functions for neurons
% tansig: bipolar sigmoidal function, with a=1 and b=2
% logsig: unipolar sigmoidal function
% purelin: linear function
hidden_layer_transfer_function = 'tansig';
output_layer_transfer_function = 'tansig';
n_layers_transfer_functions = ... % Experiment variable 3
    {hidden_layer_transfer_function, 'tansig', output_layer_transfer_function};

% TODO: what is this
% traingdx(default): gradient descent with momentum and adaptive learning
% backpropagaton. 
% trainbfg: GFGS quasi-newton backpropagation
back_propagation_training_function = 'traingd';

% The weight adjustment for any hidden layer of neurons will be derived
% using this function. See lecture note MLP slide 20. 
% traingdm(default): gradient descent with momentum backpropagation
back_propagation_weight_learning_function = 'traingdm';

% mse(default): mean square error
performance_function = 'mse';

%% create a neural network

% newff: create a feed forward, backpropagation neural networks
% net: 3 layers feed forward backpropagation network
net = newff(minmax_train_set, ...
    n_layers_number_of_neurons, ... 
    n_layers_transfer_functions, ...
    back_propagation_training_function, ...
    back_propagation_weight_learning_function, ...
    performance_function); 

%% setting up training parameters

% lr: learning rate
net.trainParam.lr = 0.1;       % Experiment variable 1

% epochs: maximum number of training iterations before training is stopped
net.trainParam.epochs = 1000;  % Experiment variable 4

% goal: performance goal
net.trainParam.goal = 0;

% max_fail: maximum number of validation checks before training is stopped
net.trainParam.max_fail = 25;

% min_grad: minimum performance gradient
net.trainParam.min_grad = 0;

% show: epochs between showing progress
net.trainParam.show = 25;

% time: maximum time to train in second
net.trainParam.time = inf;

%% Training
initial_input_delay_condition = []; % Don't care
initial_layer_delay_condition = []; % Don't care

[net, training_record] = train(net, ...
    train_set, ...
    train_set_class, ...
    initial_input_delay_condition, ...
    initial_layer_delay_condition, ...
    test_data);

%% Validation and Calculation

% fields = 1
% number_of_validation_set = 920
[fields, number_of_test_set] = size(test_set_class);

% test_set => model => results
simulation_results = sim(net, test_set);

% Since we are using bipolar sigmoidal function for output layer neuron,
% we transform the simulation results using a hard limiter.
% 1, if greater than 0
% -1, if less than 0
% 0, if equal to 0
neuralnetscore = sign(simulation_results);

% Training error rate
% TODO: check this equation
missclassification_rate = ...
    sum(0.5 * abs(test_set_class - neuralnetscore)) / number_of_test_set;

fprintf('Misclassfication rate: %f', missclassification_rate);
