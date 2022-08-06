function [ymin, ymax] = interval_bound_propagation(W, b, xmin, xmax)
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


L = size(W, 2);   % number of layers in neural network
l = 1;            % start from first layer

z_max_l = transpose(xmax);
z_min_l = transpose(xmin);

% Going through first L-1 layers of neural network
while l<L
    W_l_plus = max(0, W{l});
    W_l_minus = min(0, W{l});
    
    z_cap_max_l = W_l_plus*z_max_l + W_l_minus*z_min_l + b{l};
    z_cap_min_l = W_l_plus*z_min_l + W_l_minus*z_max_l + b{l};
    
    z_max_l = max(0, z_cap_max_l);
    z_min_l = max(0, z_cap_min_l);
    
    l = l + 1;
end

% Obtaining final results/outputs (y)
W_l_plus = max(0, W{L});
W_l_minus = min(0, W{L});

ymax = W_l_plus*z_max_l + W_l_minus*z_min_l + b{L};
ymin = W_l_plus*z_min_l + W_l_minus*z_max_l + b{L};

end