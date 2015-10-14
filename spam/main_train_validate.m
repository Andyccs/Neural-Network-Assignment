clear; clc; close all;

%% load the spam data
load('spam_data.mat');

%% Preprocessing
train_set = P_train;
train_set_class = T_train;

% Split to train_set and validation_set
% To avoid unfortunate split, we get the first 460 and last 460 from the original train set
% as validation set. Note that the class of first 460 is all -1 and the class of last 460
% is all 1. 
temp_train_set_first_460 = train_set(:, 1:460);
temp_train_set_last_460 = train_set(:, size(train_set, 2) - 459:size(train_set, 2));
validation_set = [temp_train_set_first_460, temp_train_set_last_460];

temp_train_set_class_first_460 = train_set_class(:, 1:460);
temp_train_set_class_last_460 = ...
    train_set_class(:, size(train_set_class, 2) - 459:size(train_set_class, 2));
validation_set_class = [temp_train_set_class_first_460, temp_train_set_class_last_460];

train_set = train_set(:, 461:size(train_set, 2) - 460);
train_set_class = train_set_class(:, 461:size(train_set_class, 2) - 460);

n_layers_number_of_neurons = [10, 10, 1];
n_layers_transfer_functions = {'tansig', 'tansig', 'tansig'};
learning_rate = 0.2;
iterations = 1000;

err = spam(...
    train_set,...
    train_set_class,...
    validation_set,...
    validation_set_class,...
    n_layers_number_of_neurons,...
    n_layers_transfer_functions,...
    learning_rate,...
    iterations);

fprintf('misclassification error: %f', err);