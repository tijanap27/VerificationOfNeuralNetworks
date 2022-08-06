function y = compute_nn_outputs(W, b, X)
% Compute nn outputs, which takes as its inputs the matrix X as well as the
% neural network weights and biases

% input:   W = weights  - a 1*5 vector that contains weights for 5 layers
%                               of neural network
%                                 - 40x6... 40x40... 38x40... 19x38... 1x19
%          b = biases   - a 1*5 vector that contains biases for 5 layers of
%                               neural network
%                                 - 40x1... 40x1... 38x1... 19x1... scalar
%          X = inputs   - a k*6 matrix

% output:  y = outputs  - a k*1 vector


L = size(W, 2);   % number of layers in neural network
l = 1;            % start from first layer

z = transpose(X);

% Going through first L-1 layers of neural network
while l<L
    z_cap = W{l}*z + b{l};
    z_zeros = zeros(1*size(b{l}));
    z = max(z_cap, z_zeros);
    
    l = l + 1; % moving to the next layer
end

% Obtaining final result/output (y)
y = W{L}*z + repmat(b{L}, 1, size(W{L}*z, 2));
y = transpose(y);

end