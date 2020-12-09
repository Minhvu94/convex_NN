clear
clc
close all

%% Data Processing 
load('VanderPol_traj_data.mat')
X = inputs(:,20:end)';
Y = targets(20:end)';

X_test = inputs(:,1:20)';
Y_test = targets(1:20)';


% Plot the whole data
figure 
hold on
plot(inputs(1,:),'b')
plot(inputs(2,:),'k')
plot(targets,'r')
lgd = legend('input x1','input x2','target values');
lgd.FontSize = 15;
title('The data set')
set(gca,'fontsize',14)

%% prepare D_i matrices 

[n,m] = size(X); 

% Generate random u from a uniform distribution  
U=2*(rand(m,500)-0.5);   %scatter(U(1,:),U(2,:))      

% For each u, compute a diagonal vector 
store_diag_vec = [];
for i=1:size(U,2)
    diag_vec = zeros(1,size(X,1));
    indicator = find(X*U(:,i)>=0);
    diag_vec(indicator) = 1;    
    
    if diag_vec==0
        continue
    end
    
    % Check if a diagonal vector already exist
    if  isempty(store_diag_vec) 
        store_diag_vec = [store_diag_vec; diag_vec];
    elseif ~ismember(diag_vec,store_diag_vec,'rows')
        store_diag_vec = [store_diag_vec; diag_vec];
    end
%     i
end
% store_diag_vec


% Form diagonal matrices Di
P = size(store_diag_vec,1);
for i=1:P
    D(:,:,i)=diag(store_diag_vec(i,:));
end
% D

%% Solve the regularized convex problem (8)
beta = 0.2;

cvx_begin %quiet
    variable v(m,P) 
    variable w(m,P)
    obj = 0;
    regulizer = 0;
    for i = 1:P
        obj = obj + D(:,:,i)*X*(v(:,i)-w(:,i));
        regulizer = regulizer + norm(v(:,i)) + norm(w(:,i));
    end
    obj = 0.5*norm(obj-Y)^2;
    
    minimize(obj+beta*regulizer);
    subject to
        for i = 1:P
            (2*D(:,:,i)-eye(n))*X*v(:,i)>=0;
            (2*D(:,:,i)-eye(n))*X*w(:,i)>=0;
        end
cvx_end

%% Compute optimal weights 
W1 = [];
W2 = [];
for i = 1:P
    if norm(v(:,i)) > 1e-3
        W1 = [W1, v(:,i)/sqrt(norm(v(:,i)))];
        W2 = [W2; sqrt(norm(v(:,i)))];
    end
end
for i = 1:P
    if norm(w(:,i)) > 1e-3
        W1 = [W1, w(:,i)/sqrt(norm(w(:,i)))];
        W2 = [W2; -sqrt(norm(w(:,i)))];
    end
end

%% Forward with optimal weights (using train data)   
ReLU=@(x) max(0,x);

layer1 = X*W1;
layer1_ReLU = ReLU(layer1);  
output = layer1_ReLU*W2;

training_cost = 0.5*norm(output - Y)^2 + 0.5*beta*(norm(W1,'fro')^2 + norm(W2)^2) % 0.2
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


%% Forward with optimal weights (using test data) 
layer1 = X_test*W1;
layer1_ReLU = ReLU(layer1);  
output_test = layer1_ReLU*W2;

testing_cost = 0.5*norm(output_test - Y_test)^2 + 0.5*beta*(norm(W1,'fro')^2 + norm(W2)^2) % 0.2
%% Plot testing 
figure
hold on
plot(Y_test,'r.','MarkerSize',10)
plot(output_test,'b.','MarkerSize',10)
lgd = legend('target','predict');
lgd.FontSize = 15;
title('Testing evaluation')
set(gca,'fontsize',14)
% ylim([0 3.5])









