clear, clc
%% Generate data to be used for binary classification

n = 100; % number of samples per class
mA = [ 5.0, 5.0]; sigmaA = 1.5; % x,y centers of normal distribution of class A
mB = [-5.0, -5.0]; sigmaB = 1.5; % x,y centers of normal distribution of class B

classA(1,:) = randn(1,n) .* sigmaA + mA(1); % x values class A
classA(2,:) = randn(1,n) .* sigmaA + mA(2); % y values class A

classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

% Merge class A and B into single class and shuffle
classAB = [classA, classB];
shuffle = randperm(2*n);
data = classAB(:,shuffle);
class = max(sign(shuffle-n),0); % class A: 0, class B: 1

% Plot patterns
figure(1),clf(1), hold on
scatter(classA(1,:),classA(2,:),'xr')
scatter(classB(1,:),classB(2,:), 'ob')
grid on
hold off

%% Perceptron

%% Delta learning rule
