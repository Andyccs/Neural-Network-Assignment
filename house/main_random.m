clc; clear; close all;

%% Setting up variables
for i=1 : 3

for j=1 : 10
n_layers_number_of_neurons = [j, i, 1];
n_layers_transfer_functions = {'tansig', 'tansig', 'tansig'};
learning_rate = 0.1;
iterations = 1000;

main_train_validate_random(...
  n_layers_number_of_neurons,...
  n_layers_transfer_functions,...
  learning_rate,...
  iterations);

fprintf(' j = %i , i = %i\n', j, i);

% main_train_validate_test(...
%   n_layers_number_of_neurons,...
%   n_layers_transfer_functions,...
%   learning_rate,...
%   iterations);

end
end