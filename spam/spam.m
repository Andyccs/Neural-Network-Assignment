function err = spam(...
    train_set,...
    train_set_class,...
    validation_set,...
    validation_set_class,...
    n_layers_number_of_neurons,...
    n_layers_transfer_functions,...
    learning_rate,...
    iterations)

rng('default');

%% Preprocessing

% preprocess training set by normalization to 0 mean and 1 std
[train_set, process_setting] = mapstd(train_set);
train_set_mean = process_setting.xmean;
train_set_std = process_setting.xstd;

% transform validation set by normalization to:
% 1. mean of training set
% 2. std of training set
validation_set = trastd(validation_set, train_set_mean, train_set_std);

% validation_data is a struct to hold validation_set and validation_set_class
validation_data.P = validation_set;
validation_data.T = validation_set_class;

% training set variables: train_set, train_set_class
% validation set variables: validation_set, validation_set_class, validation_data

%% Setting up variables for feed forward networks
minmax_train_set = minmax(train_set);

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
net.trainParam.lr = learning_rate;   % Experiment variable 1

% epochs: maximum number of training iterations before training is stopped
net.trainParam.epochs = iterations;  % Experiment variable 4

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
    validation_data);

%% Validation and Calculation

% fields = 1
% number_of_validation_set = 920
[~, number_of_validation_set] = size(validation_set_class);

% validation_set => model => results
simulation_results = sim(net, validation_set);

% Since we are using bipolar sigmoidal function for output layer neuron,
% we transform the simulation results using a hard limiter.
% 1, if greater than 0
% -1, if less than 0
% 0, if equal to 0
neuralnetscore = sign(simulation_results);

% Training error rate
% TODO: check this equation
missclassification_rate = ...
    sum(0.5 * abs(validation_set_class - neuralnetscore)) / number_of_validation_set;

err = missclassification_rate;

end
