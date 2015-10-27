function house_rbf(...
    train_set,...
    train_set_class,...
    validation_set,...
    validation_set_class,...
    spread,...
    number_of_neurons)

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

%% RBF
net = newrb(train_set(1:11500,:)', train_set_class(1:11500,:)', 0, spread, number_of_neurons, 1)';

y = sim(net, validation_set');
y = y';

%% Calculate misclassification errors
validation_set_class_original = poststd(validation_set_class, train_set_class_mean, train_set_class_std);
y_original = poststd(y, train_set_class_mean, train_set_class_std);
err = sqrt(mean((validation_set_class_original - y_original).^2));

fprintf('Misclassfication error: %f', err);

[validation_set_class_sorted, idx] = sort(validation_set_class_original, 1);
y_sorted = y_original(idx, :);

figure
scatter(1:size(validation_set_class_sorted), validation_set_class_sorted);
hold on
scatter(1:size(y_sorted), y_sorted, '+');

end