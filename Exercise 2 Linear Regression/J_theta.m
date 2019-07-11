clear all
x = load('ex2x.dat');
y = load('ex2y.dat');
% figure
% plot(x,y,'o');
% xlabel('Age in years'),ylabel('Height in meters');
m=length(y);
x=[ones(m,1),x];

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