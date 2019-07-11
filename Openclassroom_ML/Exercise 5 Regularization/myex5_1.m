clear all; close all; clc

% Regularization linear regression
% Using Normal Equations
x = load('ex5Linx.dat'); 
y = load('ex5Liny.dat');
x_nointercept = x;
m = length(y);

figure
plot(x_nointercept,y,'ro','MarkerFaceColor','r');hold on;

% Add intercept term to x
x = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5];
n = length(x(1,:));
lambda = [0,1,10];
theta_normal = zeros(n,length(lambda));
norm_theta = zeros(1,length(lambda));
for i=1:length(lambda)
    theta_normal(:,i) = (x' * x + lambda(i) * diag([0,ones(1,n-1)]))\x' * y;
    norm_theta(i) = norm(theta_normal(:,i));
end
theta_normal
norm_theta
x_test = linspace(-1,1,50)';
x_test = [ones(length(x_test), 1), x_test, x_test.^2, x_test.^3, x_test.^4, x_test.^5];
for i=1:length(lambda)
    plot(x_test(:,2),x_test * theta_normal(:,i),'--');
end
hold off;
legend('Training data','5th order fit,\lambda=0','5th order fit,\lambda=1','5th order fit,\lambda=10');



