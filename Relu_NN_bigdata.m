clear
clc
close all

% training data
load('VanderPol_traj_data.mat')
X = inputs(:,20:end)';
Y = targets(20:end)';

X_test = inputs(:,1:20)';
Y_test = targets(1:20)';
%%

n = size(X,1); 
m = 20; % number of hidden neurons 
beta = 0.02;
lr = 0.0001;

ReLU=@(x) max(0,x);

% for trial = 1:510

% initialize weights
W1 = rand(size(X,2),m);
W2 = rand(m,1);
store_cost = [];


for i=1:5000
% Forward
layer1 = X*W1;
layer1_ReLU = ReLU(layer1);  
output = layer1_ReLU*W2;

cost = 0.5*norm(output - Y)^2 + 0.5*beta*(norm(W1,'fro')^2 + norm(W2)^2);
store_cost = [store_cost, cost];

% Backprop
grad_loss = (output - Y);
grad_W2 = layer1_ReLU'*grad_loss + beta*(W2);
grad_layer1 = grad_loss*W2';
grad_layer1_ReLU =  gradReLU(grad_layer1,layer1);
grad_W1 = X'*(grad_layer1_ReLU) + beta*W1;

W1 = W1 - lr*grad_W1;
W2 = W2 - lr*grad_W2;

end


fprintf('min = %.6f; final = %.6f;\n',[min(store_cost) cost]);
figure
plot(store_cost,'LineWidth',2)
title('Training cost')
axis([-100 5000 -1 25])
% end

% plot optimal solution computed from the convex problem
% plot([-100 5000],[0.1977,0.1977], 'k--','LineWidth',2)

%% Plot training 
figure
hold on
plot(Y,'r.','MarkerSize',10)
plot(output,'b.','MarkerSize',6)
lgd = legend('target','predict');
lgd.FontSize = 15;
title('Training evaluation')
set(gca,'fontsize',14)
ylim([0 3.5])

%% Testing 
layer1 = X_test*W1;
layer1_ReLU = ReLU(layer1);  
output_test = layer1_ReLU*W2;

testing_cost = 0.5*norm(output_test - Y_test)^2 + 0.5*beta*(norm(W1,'fro')^2 + norm(W2)^2) % 0.2

% Plot testing 
figure
hold on
plot(Y_test,'r.','MarkerSize',10)
plot(output_test,'b.','MarkerSize',10)
lgd = legend('target','predict');
lgd.FontSize = 15;
title('Testing evaluation')
set(gca,'fontsize',14)

function x = gradReLU(x,y) 
x(find(y<0))=0;
end













