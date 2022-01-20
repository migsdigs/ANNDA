%% Generate data to be used for binary classification

n = 100; % number of samples per class
mA = [ 5.0, 5.0]; sigmaA = 0.5; % x,y centers of normal distribution of class A
mB = [-5.0, -5.0]; sigmaB = 0.5; % x,y centers of normal distribution of class B

classA(1,:) = randn(1,n) .* sigmaA + mA(1); % x values class A
classA(2,:) = randn(1,n) .* sigmaA + mA(2); % y values class A

classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

% Merge class A and B into single class and shuffle
class = 0;