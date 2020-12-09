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

n = size(X,1); 

% Generate random u from a uniform distribution  
U=2*(rand(2,100)-0.5);   %scatter(U(1,:),U(2,:))

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
end
store_diag_vec


% Form diagonal matrices Di
P = size(store_diag_vec,1);
for i=1:P
    D(:,:,i)=diag(store_diag_vec(i,:));
end
D


%% Solve the regularized convex problem (8)
beta = 0.0002;

cvx_begin %quiet
    variable v(2,P) 
    variable w(2,P)
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

% Forward NN    
ReLU=@(x) max(0,x);

layer1 = X*W1;
layer1_ReLU = ReLU(layer1);  
output = layer1_ReLU*W2;

total_cost = 0.5*norm(output - Y)^2 + 0.5*beta*(norm(W1,'fro')^2 + norm(W2)^2)












