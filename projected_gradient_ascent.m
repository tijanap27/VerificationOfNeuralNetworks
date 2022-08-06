function refined_X = projected_gradient_ascent(W, b, X, xmin, xmax)

% Compute refined X 

% input:  xmin - a 1*6 vector
%         xmax - a 1*6 vector
%         k    - integer - number of random inputs

% output: X    - a k*6 matrix - contains a valid input in each of its k rows


% Assign a random value between xmin and xmax all elements of ith row
% of the output matrix

L = size(W, 2);   % number of layers in neural network

refined_X = zeros(size(X, 1), size(X, 2));

% For each of the X_k in X
for k = 1 : size(X, 1)
    X_k = transpose(X(k, :));
    
    for iteration = 1:30
        da_dx = 0;
        propagated_da_dx = 0;
        a = X_k;
        
        % For each layer, compute the X_k to the ReLU function
        for i = 1:L
            z = W{i}*a + b{i};              % ReLU function
            
            m = size(W{i}, 1);              % Gradient matrix of ReLU function
            df_dx = zeros(m, m);
            
            % Populate the gradient matrix with 1s and 0s depending on its sign
            for j = 1:size(z, 1)
                if z(j) > 0
                    df_dx(j, j) = 1;
                else
                    df_dx(j, j) = 0;
                end
            end
            
            % Compute the gradient of the current layer
            if i<=L-1
                da_dx = df_dx * W{i};
            else
                da_dx = W{i};
            end
            
            % Compute the product of all the gradients from the previous layers
            if i==1
                propagated_da_dx = da_dx;
            else
                propagated_da_dx = da_dx * propagated_da_dx;
            end
            
            % ReLU z and set X_k to be the output of the previous layer
            if i<=L-1
                a = max(z, 0);
            end
        end
        
        nu = 0.00001;           % Learning rate
        
        % Update the X_k vector
        X_k = X_k + nu.* transpose(propagated_da_dx);
        
        % Clip X_k vector so that values lie between xmin and xmax
        X_k = min(X_k, transpose(xmax));
        X_k = max(X_k, transpose(xmin));
        
    end
    refined_X(k, :) = transpose(X_k);
end
end