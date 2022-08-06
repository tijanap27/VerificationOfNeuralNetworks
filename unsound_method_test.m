% TEST TASK 1

close all;clear;clc;
% Increase the value of k from 1 to some maximum value.

% For each value of k, plot the average amount of time (averaged over 500 properties) taken
% to generate the outputs.

% Also plot the average lower bound as well as the number of properties for
% which you were able to successfully find a counter-example.


n_properties = 500;   % number of properties

k_min = 1;            % starting value of k
k_max = 1000;         % final/max value of k

false = zeros(1, k_max);             % number of properties proven to be wrong for different k
lower_bound_sum = zeros(1, k_max);   % average lower bound for different k

for k = k_min:k_max
    k
    % loading data
    for i = 1:n_properties
        if i<10
            string = "property00" + i + ".mat";
        end
        if i>9 & i<100
            string = "property0" + i + ".mat";
        end
        if i>99
            string = "property" + i + ".mat";
        end
        data = load(string);
        variables = fields(data);
        xmin = data.xmin;
        xmax = data.xmax;
        W = data.W;
        b = data.b;
        
        tic;                                % timer on
        
        X = generate_inputs(xmin, xmax, k); % generating inputs
        y = compute_nn_outputs(W, b, X);    % obtaining outputs of neural network
        lower_bound = max(y);               % the "worst case" lower bound value obtained
        
        if lower_bound>0
            false(k) = false(k) + 1;        % if lower_bound is greater than zero, property is proven to be false
        end
        
        lower_bound_sum(k) = lower_bound_sum(k) + lower_bound;   % summing up all lower bounds over all properties
    end
    
    t = toc;                                                     % timer off
    time(k) = t/n_properties;                                    % average time
    lower_bound_sum(k) = lower_bound_sum(k)/n_properties;        % averaging lower_bound sum
end


subplot(3, 1, 1);
plot([k_min:1:k_max], time);                % plot of average time needed to obtain lower bound for different k
title('Plot of average time against k');
xlabel('k');
ylabel('Average time');

subplot(3, 1, 2);
plot([k_min:1:k_max], lower_bound_sum);     % plot of average lower bound for different k
title('Plot of average lower bound against k');
xlabel('k');
ylabel('Average lower bound');

subplot(3, 1, 3);
plot([k_min:1:k_max], false);                % plot of number of properties proven to be false for different k
title('Plot of number of number of counter-examples against k');
xlabel('k');
ylabel('Number of counter-examples');
