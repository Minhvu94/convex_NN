clear
clc
close all

% training data 
X = [-2 1;
     -1 1;
      0 1;
      1 1;
      2 1];
Y = [1;-1;1;1;-1];

% load('data.mat','X','Y')


n = size(X,1); 
m = 8; % number of hidden neurons 
beta = 0.0002;
lr = 0.01;

ReLU=@(x) max(0,x);

for trial = 1:5

% initialize weights
w1 = rand(size(X,2),m);
w2 = rand(m,1);
store_cost = [];


for i=1:5000
% Forward
layer1 = X*w1;
layer1_ReLU = ReLU(layer1);  
output = layer1_ReLU*w2;

cost = 0.5*norm(output - Y)^2 + 0.5*beta*(norm(w1,'fro')^2 + norm(w2)^2);
store_cost = [store_cost, cost];

% Backprop
grad_loss = (output - Y);
grad_w2 = layer1_ReLU'*grad_loss + beta*(w2);
grad_layer1 = grad_loss*w2';
grad_layer1_ReLU =  gradReLU(grad_layer1,layer1);
grad_w1 = X'*(grad_layer1_ReLU) + beta*w1;

w1 = w1 - lr*grad_w1;
w2 = w2 - lr*grad_w2;

end


fprintf('min = %.6f; final = %.6f;\n',[min(store_cost) cost]);

plot(store_cost,'LineWidth',2)
hold on
axis([-100 5000 -0.1 4])
end

% plot optimal solution computed from the convex problem
plot([-100 5000],[0.00202531,0.00202531], 'k--','LineWidth',2)


function x = gradReLU(x,y) 
x(find(y<0))=0;
end













