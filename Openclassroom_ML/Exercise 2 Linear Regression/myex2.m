clear all
x = load('ex2x.dat');
y = load('ex2y.dat');
figure
plot(x,y,'o');
xlabel('Age in years'),ylabel('Height in meters');
m=length(y);
x=[ones(m,1),x];
alpha = 0.07;
theta=[0;0];
theta_record = theta;
%%%%%%%%%%%%%%%%%%%%%%%%%
%这是第一次写的，有点小问题，因为在改变了theta1后就用上改变theta2
% for iteration = 1:1500
%     for j=1:length(theta)
%         temp = 0;
%         for i=1:m
%             temp = temp + (theta(1)*x(i,1)+theta(2)*x(i,2)-y(i))*x(i,j);
%         end
%         theta(j) = theta(j) - alpha/m*temp;
%     end
%     theta_record = [theta_record,theta];
%     
% end
%%%%%%%%%%%%%%%%%%%%%%%%
for iteration = 1:1500
    temp = zeros(length(theta),1);
    for j=1:length(theta)
        
        for i=1:m
            temp(j) = temp(j) + (theta(1)*x(i,1)+theta(2)*x(i,2)-y(i))*x(i,j);
        end
    end
    for n=1:length(theta)
        theta(n) = theta(n) - alpha/m*temp(n);
    end
    theta_record = [theta_record,theta];
end
hold on
plot(x(:,2),x*theta,'-');
legend('Training data','Linear Regression')
theta
%%%Prediction%%%%
age1 = [1,3.5];
age2 = [1,7];
height_predict_1 = age1 * theta;
height_predict_2 = age2 * theta;
fprintf('The boy of age 3.5,his height is %f\n',height_predict_1);
fprintf('The boy of age 7.0,his height is %f\n',height_predict_2);

theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
J_vals = zeros(length(theta0_vals), length(theta1_vals));   % initialize Jvals to 100x100 matrix of 0's
for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = (0.5/m).*(sum((x*t-y).^2));
    end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')
% to see the approach to the global optimum more apparent
figure;
% Plot the cost function with 15 contours spaced logarithmically
% between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 2, 15));
xlabel('\theta_0'); ylabel('\theta_1');