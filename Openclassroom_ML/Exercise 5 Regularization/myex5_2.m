clear all; close all; clc

% Regularization logistic regression
% Using Newton's Method
x = load('ex5Logx.dat'); 
y = load('ex5Logy.dat');
m = length(y);
x_expand = map_feature(x(:,1),x(:,2));

% Find the indices for the 2 classes
pos = find(y); neg = find(y == 0);

% plot(x(pos, 1), x(pos, 2), 'k+','LineWidth',1.2)
% hold on
% plot(x(neg, 1), x(neg, 2), 'ko','MarkerFaceColor','y')
% xlabel('u');ylabel('v');legend('y=1','y=0');hold off;

% Newton's Method Iterations
g = inline('1.0 ./ (1.0 + exp(-z))'); 
MAX_ITR = 15;
theta = zeros(size(x_expand(1,:)))'; % initialize fitting parameters
lambda = [0,1,10];
J = zeros(MAX_ITR, length(lambda));

for choose_lambda = 1:length(lambda)
    for num_iterations = 1:MAX_ITR
        % Calculate coss J, vectorized implement
        G = g(x_expand * theta);
        G1 = 1 - G;
        S = log(G);
        V = log(G1);
        % Regularized logistic regression cost function J
        % Add the regularization term
        J(num_iterations,choose_lambda) = (-1.0/m) .* (y' * S + (1 - y)' * V) + (lambda(choose_lambda)/(2*m)).* (theta(2:end)' * theta(2:end)); 

        % Update theta
        grad_J_before = (1/m) .* x_expand' * (G - y); % J gradient
        extra_theta = [0;(lambda(choose_lambda)/m) .* theta(2:end)];
        grad_J = grad_J_before + extra_theta;
        H = 0; % Hessian matrix initial
        for i = 1:m
            H = H + (1/m) .* G(i) * G1(i) .* (x_expand(i,:)' * x_expand(i,:));
        end
        H = H + (lambda(choose_lambda)/m) .* diag([0,ones(1,length(theta)-1)]);
        theta = theta - H \ grad_J; % use Newton's Method to update theta

    end
    norm_theta(choose_lambda) = norm(theta);
    % Plot decision boundary 
    % Define the ranges of the grid
    u = linspace(-1, 1.5, 200);
    v = linspace(-1, 1.5, 200);

    % Initialize space for the values to be plotted
    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            % Notice the order of j, i here!
            z(j,i) = map_feature(u(i), v(j))*theta;
        end
    end

    % Because of the way that contour plotting works
    % in Matlab, we need to transpose z, or
    % else the axis orientation will be flipped!
    z = z';
    % Plot z = 0 by specifying the range [0, 0]
    figure
    plot(x(pos, 1), x(pos, 2), 'k+','LineWidth',1.2)
    hold on
    plot(x(neg, 1), x(neg, 2), 'ko','MarkerFaceColor','y')
    xlabel('u');ylabel('v');
    contour(u,v,z, [0, 0], 'LineWidth', 2)
    legend('y = 1', 'y = 0', 'Decision boundary');
    hold off;
    title(sprintf('\\lambda = %g', lambda(choose_lambda)), 'FontSize', 14);
end
J
norm_theta
fprintf('Want to see detailed value,go to workspace!\n')
