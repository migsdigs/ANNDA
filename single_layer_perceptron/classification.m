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
axis([-10,10,-10,10])
axis square
scatter(classA(1,:),classA(2,:),'xr')
scatter(classB(1,:),classB(2,:), 'ob')
grid on
hold off

%% Perceptron

%% Delta learning rule
eta = 0.0001; % Learning rate
epoch = 100;
W = randn(2,1); % Initialise weights using standard normal

figure(1),
h1 = animatedline;
h2 = animatedline;
for epoch = 1:epoch
    dw = 0; % Reset dw
    for i = 1:(2*n)
        e = class(i) - W'*data(:,i);
        dw = dw + eta*e*data(:,i); % Accumulate dw
    end
    W = W + dw; % Apply change
    
    % Plotting intermediate results
    clearpoints(h1)
    clearpoints(h2)
    addpoints(h1,[0 W(1)/norm(W)],[0,W(2)/norm(W)]);
    drawnow
    addpoints(h2,[10*W(2)/W(1),-10*W(2)/W(1)],[-10,10])
    drawnow
    pause(0.05)
end