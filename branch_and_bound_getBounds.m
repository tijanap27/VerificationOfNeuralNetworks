function bounds = branch_and_bound_getBounds(W, b, xmin, xmax)

% Takes the neural network weights and biases as well as the upper and
% lower bounds of the input, and returns the upper and lower bounds of the
% output.

% input:  xmin - a 1*6 vector
%         xmax - a 1*6 vector
%         W = weights  - a 1*5 vector that contains weights for 5 layers
%                               of neural network
%                                 - 40x6... 40x40... 38x40... 19x38... 1x19
%         b = biases   - a 1*5 vector that contains biases for 5 layers of
%                               neural network
%                                 - 40x1... 40x1... 38x1... 19x1... scalar

% output: ymax - upper bound of the output
%         ymin - lower bound of the output

k = 200;

X = generate_inputs(xmin, xmax, k);
y = compute_nn_outputs(W, b, X);
lower_max = max(y);
upper_min = min(y);

[lower_max_lp, upper_min_lp] = interval_bound_propagation(W, b, xmin, xmax);

bounds = [lower_max_lp, upper_min, lower_max, upper_min_lp];
end