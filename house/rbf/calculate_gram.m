function gram = calculate_gram(data_set, centroids, correlation);
  number_of_inputs = size(data_set, 1);
  number_of_cluster = size(centroids, 1);

  %% Calculate Gram Matrix
  gram = zeros(number_of_inputs, number_of_cluster);

  for i = 1 : number_of_cluster
      for j = 1 : number_of_inputs
          distance = data_set(j,:) - centroids(i, :);
          gram(j, i) = exp( -0.5 .* (distance * correlation{i} * distance') );
      end
  end

end