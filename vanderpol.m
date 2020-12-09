clear
clc
close all

syms xi1 xi2 xi3 dt

F = [xi2;
     -xi1+xi2-xi1^2*xi2+xi3;
      0];
  
F = simplify(F);
     
p = 6;
tic
[Phi,Psi_p,JPhi] = compute_Phi_and_JPhi(p,F,[xi1 xi2 xi3],dt);
%%
T = 2;  
dt = 0.01;
iter_max = ceil(T/dt);
x0 = [3;3];
x_target = [0;0];
u = zeros(iter_max,1);
lambda = 0.001;



while true
    x = x0;
    x_traj = [];
    for iter = 1:iter_max
        % Prepare A,B to later calculate H
        R_big = JPhi(dt,x(1),x(2),u(iter));        
        R_store{iter} = R_big(1:2,1:2);       % A in Ax+Bu 
        B_store{iter} = R_big(1:2,3:3);       % B in Ax+Bu
        % The actual trajectory using adaptive step size mechanism
        [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u(iter)]); 
        x = x_trajJ_fine(end,:)'; % the end of this sequence is x[k+1]
        x = x(1:2);
        x_traj = [x_traj x];      
    end    
    error = norm(x-x_target);
    if error <= 0.001
        error
        break
    end
    % Calculate H
    H = B_store{1};
    for iter = 2:iter_max
        H = [R_store{iter}*H, B_store{iter}];
    end
    u = u - (H'*H+lambda*error*eye(iter_max))\(H'*(x-x_target));
end 

Fig = openfig('vanderpol.fig');
plot(x_traj(1,:),x_traj(2,:),'r','LineWidth',2)

inputs = x_traj(:,1:end-1);
targets = x_traj(1,2:end);

save('VanderPol_traj_data.mat','inputs','targets')

function [u,x_traj] = part2(p,Phi,Psi_p,JPhi,dt,iter_max,x0,x_target,u)
old_cost = norm(u);
mu = 5;
for iter2 = 1:500       
    x = x0;    
    x_traj = [];
    for iter = 1:iter_max
        % Prepare A,B to later calculate H
        R_big = JPhi(dt,x(1),x(2),u(iter));        
        R_store{iter} = R_big(1:2,1:2);       % A in Ax+Bu 
        B_store{iter} = R_big(1:2,3:3);       % B in Ax+Bu

        % The actual trajectory using adaptive step size mechanism
        [~, x_trajJ_fine] = adaptive_taylor(p,Phi,Psi_p,[0 dt],[x;u(iter)]); 
        x = x_trajJ_fine(end,:)'; % the end of this sequence is x[k+1]
        x = x(1:2);
        x_traj = [x_traj x];         
    end
    % create H:
    H = B_store{1};
    for iter = 2:iter_max
        H = [R_store{iter}*H, B_store{iter}];       
    end
    options = optimset('display','off');
    u = u + quadprog((1+mu)*eye(iter_max),2*u,[],[],H,x_target-x,[],[],[],options);
    improve = old_cost - norm(u);
    old_cost = norm(u);
    if improve<1e-4
        break 
    end
end

end

