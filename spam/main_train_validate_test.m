clear; clc; close all;

%% load the spam data
load('spam_data.mat');

%% Preprocessing
train_set = P_train;
train_set_class = T_train;

test_set = P_test;
test_set_class = T_test;

% Split to train_set and validation_set
% To avoid unfortunate split, we get the first 460 and last 460 from the original train set
% as validation set. Note that the class of first 460 is all -1 and the class of last 460
% is all 1. 

n_layers_number_of_neurons = [10, 10, 1];
n_layers_transfer_functions = {'tansig', 'tansig', 'tansig'};
learning_rate = 0.1;
iterations = 1000;

err = spam(...
    train_set,...
    train_set_class,...
    test_set,...
    test_set_class,...
    n_layers_number_of_neurons,...
    n_layers_transfer_functions,...
    learning_rate,...
    iterations);

fprintf('misclassification error: %f', err);