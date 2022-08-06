function X = generate_inputs(xmin, xmax, k)
% Generate k random inputs that lie in the input domain. Compute the output
% for each of the k random inputs.

% input:  xmin - a 1*6 vector
%         xmax - a 1*6 vector
%         k    - integer - number of random inputs

% output: X    - a k*6 matrix - contains a valid input in each of its k rows


% Assign a random value between xmin and xmax all elements of ith row
% of the output matrix

for i = 1:k
    X(i, :) = xmin + (xmax - xmin).*rand(1, 6);
end
end