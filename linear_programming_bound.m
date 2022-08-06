function [ymin, ymax] = linear_programming_bound(W, b, xmin, xmax)

% A function that linearises constraints for each layer in order to get
% better upper and lower bounds
%
% input:  xmin - a 1*6 vector
%         xmax - a 1*6 vector
%         W = weights  - a 1*5 vector that contains weights for 5 layers
%                               of neural network
%                                 - 40x6... 40x40... 38x40... 19x38... 1x19
%         b = biases   - a 1*5 vector that contains biases for 5 layers of
%                               neural network
%                                 - 40x1... 40x1... 38x1... 19x1... scalar
%         k    - integer - number of random inputs

% output: ymax - upper bound of the output
%         ymin - lower bound of the output


L = size(W, 2);     % number of layers in neural network 

% calculating constraints for layer 1
old_zmax = transpose(xmax);
old_zmin = transpose(xmin);

n_layers_prev = size(old_zmax, 1);  % number of nuerons in previous layer (b_size)
n_layers = size(b{1}, 1);           % number of nuerons in layer (n_layers)

new_zmax = zeros(1, n_layers);
new_zmin = zeros(1, n_layers);

e = 0.000001;

% initialise equality matrices
A = [1, zeros(1, n_layers_prev); ...
    -1, zeros(1, n_layers_prev); ...
    zeros(n_layers_prev, 1), eye(n_layers_prev); ...
    zeros(n_layers_prev, 1), -eye(n_layers_prev)]; 

B = [1+e, -1+e, transpose(old_zmax), -transpose(old_zmin)];   


% looping through current neurons (layer 1)
for n = 1:n_layers        
    f = [b{1}(n), W{1}(n, :)];
    
    x = linprog(f, A, B);
    new_zmin(n) = W{1}(n, :)*x(2 : 1+n_layers_prev) + b{1}(n);
    
    x = linprog(-f, A, B);
    new_zmax(n) = W{1}(n, :)*x(2 : 1+n_layers_prev) + b{1}(n);
end


% going through other layers
gap = 0;
for l = 2:L
    n_layers = size(b{l}, 1);           % number of nuerons in current layer
    n_layers_prev = size(b{l-1}, 1);    % number of neurons in previous layer
    b_size = size(b{l}, 1);             % size of b
    
    old_zmin = new_zmin;
    old_zmax = new_zmax;
    new_zmax = zeros(1, n_layers);
    new_zmin = zeros(1, n_layers);
    
    %calculate gradient and intercept
    gradient = old_zmax./(old_zmax - old_zmin); % this is not really gradient, more like slope of the upper bound but it's eaiser for me to call it gradient
    intercept = -gradient.*old_zmin;
    

    k_low = find(gradient>=1);
    if nnz(k_low)>=1
        gradient(k_low) = 1;
        intercept(k_low) = e;
    end

    k_high = find(old_zmax<=e);
    if nnz(k_high)>=1
        gradient(k_high) = 0;
        intercept(k_high) = e;
    end
    
    [rows_A, columns_A] = size(A);      % current size of the A matrix
    
    % Adding the new constraints into the matrix A
    if l<(L-1)
        A = [A, zeros(rows_A, n_layers_prev);...
            b{l-1}, zeros(n_layers_prev, gap), W{l-1}, -eye(n_layers_prev) ;... 
            -transpose(gradient).*b{l-1}, zeros(n_layers_prev, gap), -transpose(gradient).*W{l-1}, eye(n_layers_prev);...
            zeros(n_layers_prev, columns_A), -eye(n_layers_prev)];  
    else
        A = [A, zeros(rows_A, n_layers_prev);...
            b{l-1}, zeros(n_layers_prev, gap), W{l-1}, -eye(n_layers_prev) ;... 
            -transpose(gradient).*b{l-1},zeros(n_layers_prev, gap), -transpose(gradient).*W{l-1}, eye(n_layers_prev);...
            zeros(n_layers_prev, columns_A), -eye(n_layers_prev)];      
    end
    
    B = [B, zeros(1, n_layers_prev), intercept, zeros(1, n_layers_prev)];   
    [new_rows, new_columns] = size(A);   % updating the size of A
    
    for n = 1:b_size     % current neuron number
        f = [b{l}(n), zeros(1, columns_A-1), W{l}(n, :)];
        
        x = linprog(f, A, B);
        new_zmin(n) = W{l}(n,:)*x(new_columns-n_layers_prev+1 : new_columns) + b{l}(n);
        
        x = linprog(-f, A, B);
        new_zmax(n) = W{l}(n, :)*x(new_columns-n_layers_prev+1 : new_columns) + b{l}(n);
    end
    
    gap = size(A, 2) - 1 - n_layers_prev;
    
end

ymin = new_zmin;
ymax = new_zmax;

end
