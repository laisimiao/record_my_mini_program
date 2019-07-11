clear all;close all; clc
x = load('ex3x.dat');
y = load('ex3y.dat');
m = length(y);
x = [ones(m,1),x];xx=x;yy=y;
% 因为x数据scale不一，对其归一化处理
sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

MAX_ITR = 100;
%% alpha = 0.01 
theta = zeros(size(x(1,:)))'; % initialize fitting parameters
alpha = 0.01;   %% My initial learning rate %%
J = zeros(MAX_ITR, 1); 
%% alpha = 0.03 
theta1 = zeros(size(x(1,:)))';
alpha1 = 0.03;   
J1 = zeros(MAX_ITR, 1); 
%% alpha = 0.1 
theta2 = zeros(size(x(1,:)))';
alpha2 = 0.1;   
J2 = zeros(MAX_ITR, 1); 
%% alpha = 0.3 
theta3 = zeros(size(x(1,:)))';
alpha3 = 0.3;   
J3 = zeros(MAX_ITR, 1); 
%% alpha = 1 
theta4 = zeros(size(x(1,:)))';
alpha4 = 1;   
J4 = zeros(MAX_ITR, 1); 
%% alpha = 1.3 
theta5 = zeros(size(x(1,:)))';
alpha5 = 1.3;   
J5 = zeros(MAX_ITR, 1); 
%% alpha = 1.4 
theta6 = zeros(size(x(1,:)))';
alpha6 = 1.4;   
J6 = zeros(MAX_ITR, 1); 


for num_iterations = 1:MAX_ITR
    %% alpha = 0.01 
    J(num_iterations) = (0.5/m).*(x * theta - y)'*(x * theta - y);%% Calculate my cost function  %%
    grad = (1/m).* x' * ((x * theta) - y);
    theta = theta - alpha .* grad; %% Result of gradient descent update %%
    %% alpha = 0.03 
    J1(num_iterations) = (0.5/m).*(x * theta1 - y)'*(x * theta1 - y);
    grad1 = (1/m).* x' * ((x * theta1) - y);
    theta1 = theta1 - alpha1 .* grad1; 
    %% alpha = 0.1 
    J2(num_iterations) = (0.5/m).*(x * theta2 - y)'*(x * theta2 - y);
    grad2 = (1/m).* x' * ((x * theta2) - y);
    theta2 = theta2 - alpha2 .* grad2; 
    %% alpha = 0.3 
    J3(num_iterations) = (0.5/m).*(x * theta3 - y)'*(x * theta3 - y);
    grad3 = (1/m).* x' * ((x * theta3) - y);
    theta3 = theta3 - alpha3 .* grad3; 
    %% alpha = 1 
    J4(num_iterations) = (0.5/m).*(x * theta4 - y)'*(x * theta4 - y);
    grad4 = (1/m).* x' * ((x * theta4) - y);
    theta4 = theta4 - alpha4 .* grad4; 
    %% alpha = 1.3
    J5(num_iterations) = (0.5/m).*(x * theta5 - y)'*(x * theta5 - y);
    grad5 = (1/m).* x' * ((x * theta5) - y);
    theta5 = theta5 - alpha5 .* grad5;
    %% alpha = 1.4 
    J6(num_iterations) = (0.5/m).*(x * theta6 - y)'*(x * theta6 - y);
    grad6 = (1/m).* x' * ((x * theta6) - y);
    theta6 = theta6 - alpha6 .* grad6;
end
fprintf('Finally,gradient descent with alpha= %.1f,after %d iterations,get:\n',alpha4,MAX_ITR);
final_theta = theta4
pdc_obj = [1,1650,3];
pdc_obj(2) = (pdc_obj(2) - mu(2))./ sigma(2);
pdc_obj(3) = (pdc_obj(3) - mu(3))./ sigma(3);
prediction = pdc_obj * final_theta
% Using normal equations to calculate theta:
fprintf('Finally,using normal equations,get:\n');
NE_theta = inv(xx'*xx)*(xx')*yy;
NE_theta
prediction = [1,1650,3] * NE_theta
% now plot J
% technically, the first J starts at the zero-eth iteration
% but Matlab/Octave doesn't have a zero index
figure;
plot(0:49, J(1:50), 'b-','LineWidth',2); %% alpha = 0.01
xlabel('Number of iterations');
ylabel('Cost J');
hold on;
plot(0:49, J1(1:50), 'r-','LineWidth',2); %% alpha = 0.03
plot(0:49, J2(1:50), 'g-','LineWidth',2); %% alpha = 0.1
plot(0:49, J3(1:50), 'k-','LineWidth',2); %% alpha = 0.3
plot(0:49, J4(1:50), 'b--','LineWidth',2); %% alpha = 1
plot(0:49, J5(1:50), 'r--','LineWidth',2); %% alpha = 1.3
legend('0.01', '0.03','0.1','0.3','1','1.3');
hold off;
figure;
plot(0:49, J6(1:50), 'r--','LineWidth',2); %% alpha = 1.4
xlabel('Number of iterations');
ylabel('Cost J');
legend('1.4');