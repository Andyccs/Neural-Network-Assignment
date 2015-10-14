function plotData(X, Y)
X = X';
Y = Y';

for i=1 : size(X, 2)
  class1 = find(Y==-1);
  class2 = find(Y==1);

  for j=1 : size(X, 2)
    figure;
    hold on;
    fprintf('printing %i %i', i, j)
    scatter(X(class1,i), X(class1,j), 10, 'r', '+');
    scatter(X(class2,i), X(class2,j), 10, 'g', 'o', 'filled');
    hold off;
    pause
    close all;
  end

end

end
