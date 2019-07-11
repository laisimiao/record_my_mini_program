clear all; close all; clc
% in this code, almost use vectorized implement
x = load('ex4x.dat'); 
y = load('ex4y.dat');

m = length(y);

% Add intercept term to x
x = [ones(m, 1), x];

% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);

% Assume the features are in the 2nd and 3rd
% columns of x
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'ro');
xlabel('Exam 1 score');
ylabel('Exam 2 score');


% To define sigmoid function through an inline expression:
g = inline('1.0 ./ (1.0 + exp(-z))'); 
% Usage: To find the value of the sigmoid 
% evaluated at 2, call g(2),z can be a vector.

MAX_ITR = 7;
theta = zeros(size(x(1,:)))'; % initialize fitting parameters
J = zeros(MAX_ITR, 1);

for num_iterations = 1:MAX_ITR
    % calculate coss J, vectorized implement
    G = g(x * theta);
    G1 = 1 - G;
    S = log(G);
    V = log(G1);
    J(num_iterations) = (-1.0/m) .* (y' * S + (1 - y)' * V); % logistic regression cost function J
    
    % update theta
    grad_J = (1/m) .* x' * (G - y); % J gradient
    H = 0; % Hessian matrix initial
    for i = 1:m
        H = H + (1/m) .* G(i) * G1(i) .* (x(i,:)' * x(i,:));
    end
    theta = theta - inv(H) * grad_J; % use Newton's Method to update theta
end
theta
pro = 1-g([1,20,80] * theta);
fprintf('the probability that a student with a score of 20 on Exam 1 and a score of 80 on Exam 2 will not be admitted is:%.3f\n',pro);
% plot the decision boundary line
plot(x(:,2),-((theta(1)*x(:,1)+theta(2)*x(:,2))/(theta(3))));
xlim([10,70]);ylim([40,100]);
legend('Admitted','Not admitted','Decision boundary');
hold off;
figure;
plot(0:MAX_ITR-1,J,'b--');hold on;
plot(0:MAX_ITR-1,J,'r*');
xlabel('Iteration');ylabel('J');