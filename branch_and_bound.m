function flag = branch_and_bound(W, b, xmin, xmax)

% Takes the neural network weights and biases and upper and lower bounds of
% the input, and returns a Boolean flag indicating whether the property is true or false.

% input:  xmin - a 1*6 vector
%         xmax - a 1*6 vector
%         W = weights  - a 1*5 vector that contains weights for 5 layers
%                               of neural network
%                                 - 40x6... 40x40... 38x40... 19x38... 1x19
%         b = biases   - a 1*5 vector that contains biases for 5 layers of
%                               neural network
%                                 - 40x1... 40x1... 38x1... 19x1... scalar

% output: flag - boolean - indicates whether the property is true or false


% check if the property is true based on its initial upper bound
%       if yes: terminate
%       if no: continue

% initial x_bar_min and x_bar_max are set to input values
x_bar_min = transpose(xmin);
x_bar_max = transpose(xmax);

% bounds of starting box constraints (xmin and xmax)
bounds = branch_and_bound_getBounds(W, b, xmin, xmax);

% check if the lower bound is greater than zero and return false if yes
if (bounds(1)>0)
    flag = 0;
else
    bounds = transpose(bounds);
    
    small = 0.00001;        % constant that determines termination with flag = 2
    
    while true
        [value, index] = max(bounds(4, :));     % finding parition with maximum upper bound
        
        % if upper bound is less than zero, return true
        if value <= 0
            flag = 1;
            break;
        end
        
        % caculate relative length in ith dimension and find index of maximum value
        split = (x_bar_max(:, index) - x_bar_min(:, index)) .* 1./transpose(max(xmax - xmin, 0.00001));
        [~, splitting_index] = max(split);
        
        % set up the vectors for two new partitions
        x_bar_min_1 = x_bar_min(:, index);
        x_bar_max_1 = x_bar_max(:, index);
        x_bar_min_2 = x_bar_min(:, index);
        x_bar_max_2 = x_bar_max(:, index);
        
        % check rate of convergence - if its too small, terminate with flag = 2
        gap = (x_bar_max_1(splitting_index) - x_bar_min_1(splitting_index));
        if (gap <= small)
            break;
        end
        mid = x_bar_min_1(splitting_index) + gap/2;
        
        % partition an element of the input domain into two
        x_bar_max_1(splitting_index) = x_bar_min_1(splitting_index) + gap/2;
        x_bar_min_2(splitting_index) = x_bar_min_1(splitting_index) + gap/2;
        
        
        % if too many partitions terminate with flag = 2
        if index > 2000
            flag = 2;
            return
        end
        
        % find bounds and check for termination for new partitions
        bounds_1 = branch_and_bound_getBounds(W, b, transpose(x_bar_min_1), transpose(x_bar_max_1));
        bounds(:, index) = transpose(bounds_1);
        x_bar_min(:, index) = x_bar_min_1;
        x_bar_max(:, index) = x_bar_max_1;
        
        if (bounds_1(3)>0)
            flag = 0;
            break;
        end
        
        [~, columns] = size(bounds);
        columns = columns + 1;
        
        bounds_2 = branch_and_bound_getBounds(W, b, transpose(x_bar_min_2), transpose(x_bar_max_2));
        bounds(:, columns) = transpose(bounds_2);
        x_bar_min(:, columns) = x_bar_min_2;
        x_bar_max(:, columns) = x_bar_max_2;
        
        if (bounds_2(3)>0)
            flag = 0;
            break;
        end
    end
end
end