% TEST TASK 2


close all;clear;clc;
% Report the average values of the upper and lower bounds computed using
% interval bound propagation over all 500 properties.

% Report the number of properties that you were able to prove using
% interval bound propagation.

n_properties = 500;   % number of properties

k_min = 1;            % starting value of k
k_max = 1;         % final/max value of k

lower_bound_sum = zeros(1, k_max);   % average lower bound for different k using interval bound propagation
upper_bound_sum = zeros(1, k_max);   % average upper bound for different k using interval bound propagation

true = zeros(1, k_max);              % number of properties proven to be true for different k
false = zeros(1, k_max);             % number of properties proven to be false for different k

for k = k_min:k_max
    tic;                             % timer on
    
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
        
        tic;
        
        [ymin, ymax] = interval_bound_propagation(W, b, xmin, xmax);
        lower_bound = max(ymin);            % lower bound obtained using interval bound propagation
        upper_bound = max(ymax);            % upper bound obtained using interval bound propagation
        
        if upper_bound<=0
            true(k) = true(k) + 1;           % if upper bound is nonnegative, property is proven to be true
        end
        
        if lower_bound>0
            false(k) = false(k) + 1;        % if lower_bound is greater than zero, property is proven to be false
        end
        
        lower_bound_sum(k) = lower_bound_sum(k) + lower_bound;    % summing up all lower bounds over all properties
        upper_bound_sum(k) = upper_bound_sum(k) + upper_bound;    % summing up all upper bounds over all properties
        
    end
    
    t = toc;                                                     % timer off
    time(k) = t/n_properties;                                                 % average time
    lower_bound_sum(k) = lower_bound_sum(k)/500;                 % averaging lower_bound sum
    upper_bound_sum(k) = upper_bound_sum(k)/500;                 % averaging upper_bound sum
end

mean(lower_bound_sum)                       % print lower bound obtained
mean(upper_bound_sum)                       % print upper bound obtained
mean(true)                                  % print number of true properties
mean(false)                                 % print number of false properties
mean(time)

subplot(1, 1, 1);
plot([k_min:1:k_max], time);                % plot of average time needed to obtain lower bound for different k
title('Plot of average time against k');
xlabel('k');
ylabel('Average time');
