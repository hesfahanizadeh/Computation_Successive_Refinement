% Project: Streaming Iterative distributed computing
% Author: Homa Esfahanizadeh, Alejandro Cohen, Muriel Médard
% Last modified: 2022/07/29
% Goal: Delay vs computational redundancy for layered-resolution computations

clc
clear
close all

%% Parameter Setting

J = 10000; % number of jobs
lambda = 0.01; % arrival rate of jobs, in [job/slots]
Z = 50000; % computational complexity of a job iteration
K = 1000; % number of critical tasks per job
m = 2; % The layering parameter that shows each number is split into how many partitions
L = 2 * m - 1; % number of computational layers
max_Omega = 1.1; % max redundancy ratio
gamma = 1; % relative importance of the first moment and the second moment
P = 5; % number of workers
rng(900) % a random seed -- to realize the same workers and job arrivals 
interarrival_time = exprnd(1/lambda,1,J); % interrival time of the jobs, exponentially distributed
mu_vec = rand(1,P);
mu_vec = (Z*max_Omega*lambda) * (4*mu_vec/sum(mu_vec)); % computing rate
arrival_time_vec = cumsum(interarrival_time); % job arrival time

Omega_vec = 1:0.002:max_Omega;

delay_sim = zeros (1,length(Omega_vec));
delay_sim_layer = zeros (L,length(Omega_vec));

num_samples = 1000;
samples = zeros(L,num_samples,length(Omega_vec));

% no layering
C = Z / K; % computational complexity of each mega task per job
m_vec = C./ mu_vec;
sigma_vec = C./mu_vec;
for ind = 1:length(Omega_vec)
    disp(ind);
    Omega = Omega_vec(ind);

    start_comp_vec = zeros(1,J); % assignment processing start per job
    end_time_vec = zeros (1,J); % job end time (K tasks are finished)

    for j = 1:J
        
        if ( j==1 )
            start_comp_vec(j) = arrival_time_vec(j);
        else
            start_comp_vec(j) = max(end_time_vec(j-1),arrival_time_vec(j));
        end
        
        [kappa_vec,theta] = optimal_load_split ( gamma, zeros(1,P), m_vec , sigma_vec , K , Omega );
        kappa_vec = round (kappa_vec);
        
        while (sum(kappa_vec)<(K*Omega))
            rand_ind = randi(P);
            kappa_vec(rand_ind) = kappa_vec(rand_ind) + 1;
        end
        while (sum(kappa_vec)>(K*Omega))
            rand_ind = randi(P);
            if (kappa_vec(rand_ind)>0)
                kappa_vec(rand_ind) = kappa_vec(rand_ind) - 1;
            end
        end

        job_progress = zeros(1,ceil(K*Omega)); % end time of each task per job
        
        for p = 1:P
            tasks_time = exprnd(C/mu_vec(p),1,kappa_vec(p));
            job_progress(sum(kappa_vec(1:p-1))+1:sum(kappa_vec(1:p))) = cumsum(tasks_time);
        end
        job_progress = sort(job_progress);
        end_time_vec(j) = start_comp_vec(j)+job_progress(K); % purging
    end
    samples_ind = end_time_vec-arrival_time_vec;
    delay_sim(1,ind) = mean (samples_ind);
end

% layering
J_mini = [1:m,m-1:-1,1]; % number of mini-jobs (matrix-matrix multiplications) per layer
C_mini = C / m / m; % computational complexity of each mini task
m_mini_vec = C_mini ./ mu_vec;
sigma_mini_vec = C_mini ./ mu_vec;


for ind = 1:length(Omega_vec)
    disp(ind);
    Omega = Omega_vec(ind);
    
    start_comp_vec = zeros(1,J); % assignment processing start per job
    end_time_vec = zeros (J,L); % job layer end time (K tasks are finished)
    
    for j = 1:J
        if ( j==1 )
            start_comp_vec(j) = arrival_time_vec(j);
        else
            start_comp_vec(j) = max(end_time_vec(j-1,L),arrival_time_vec(j));
        end
        
        for l = 1:L
            for j_mini = 1:1:J_mini(l) % This layer consists of j_mini matrix-matrix multiplications 
                [kappa_vec,theta] = optimal_load_split ( gamma, zeros(1,P), m_mini_vec , sigma_mini_vec , K , Omega );
                kappa_vec = round (kappa_vec); 

                while (sum(kappa_vec)<(K*Omega))
                    rand_ind = randi(P);
                    kappa_vec(rand_ind) = kappa_vec(rand_ind) + 1;
                end
                
                while (sum(kappa_vec)>(K*Omega))
                    rand_ind = randi(P);
                    if (kappa_vec(rand_ind)>0)
                        kappa_vec(rand_ind) = kappa_vec(rand_ind) - 1;
                    end
                end 
            
                job_progress = zeros(1,ceil(K*Omega)); % end time of each mini task
            
                for p = 1:P
                    tasks_time = exprnd(C_mini/mu_vec(p),1,kappa_vec(p));
                    job_progress(sum(kappa_vec(1:p-1))+1:sum(kappa_vec(1:p))) = cumsum(tasks_time);
                end
                job_progress = sort(job_progress);
                
                if (l==1)
                    end_time_vec(j,l) = start_comp_vec(j)+job_progress(K); % purging
                else
                    if (j_mini == 1 )
                        end_time_vec(j,l) = end_time_vec(j,l-1)+job_progress(K);
                    else
                        end_time_vec(j,l) = end_time_vec(j,l)+job_progress(K);
                    end
                end
            end
        end
    end
    
    for l=1:L
        samples_ind = end_time_vec(:,l)-arrival_time_vec';
        delay_sim_layer(l,ind) = mean (samples_ind);
        samples(l,:,ind) = samples_ind(1:num_samples);
    end
end




% theory lower bound no layering
SS1 = Z ./sum(mu_vec);
SS2 = SS1^2;
rho = lambda .* SS1;
delay_LB = ((lambda.*SS2)./(2*(1-rho))) + SS1;


% theory lower with layering
delay_LB_layer = zeros(1,L);
Z_layer = (cumsum(J_mini) /sum(J_mini) )*Z;
for l=1:L
    SS1 = Z ./sum(mu_vec);
    SS1_prime = Z_layer(l) ./sum(mu_vec);
    SS2 = SS1^2;
    rho = lambda .* SS1;
    delay_LB_layer(l) = ((lambda.*SS2)./(2*(1-rho))) + SS1_prime;
end

figure;
p = {};
s = {};
%plot(Omega_vec,delay_LB*ones(size(Omega_vec)),'k:','linewidth',1.5);
hold on
pp = plot(Omega_vec,delay_sim(1,:),'ks','linewidth',1);
p = [p,pp];
s = [s,'No layering'];
cc = hsv(L); % creates colormap
cc(2,:)=[0,0.5,0];
for l=1:L
    plot(Omega_vec,delay_LB_layer(l)*ones(size(Omega_vec)),':','color',cc(l,:),'linewidth',2);
    pp = plot(Omega_vec,delay_sim_layer(l,:),'-','color',cc(l,:),'linewidth',2);
    p = [p,pp];
    s = [s,['Layer  ' num2str(l-1)]];
end
grid on
xlabel('Redundancy ratio','fontsize',30);
ylabel('Average execution delay','fontsize',30);
xlim([1,max_Omega])
set(gca,'fontsize',24,'fontname','Times New Roman') % Sets the width of the axis lines, font size, font
h = legend( p, s);
set(h,'FontSize',24);
h.Position(1) = .65 - h.Position(3)/2;
h.Position(2) = .43 - h.Position(4)/2;

figure;
p = {};
s = {};
hold on
for l=1:L
    pp = plot(Omega_vec,delay_sim_layer(l,:),'color',cc(l,:),'linewidth',2);
    plot(Omega_vec,reshape(samples(l,1:100,:),100,length(Omega_vec))','.','color',cc(l,:),'linewidth',1.5);
    p = [p,pp];
    s = [s,['Layer  ' num2str(l-1)]];
end
grid on
xlabel('Redundancy ratio','fontsize',30);
ylabel('Delay realizations','fontsize',30);
xlim([1,max_Omega])
set(gca,'fontsize',24,'fontname','Times New Roman') % Sets the width of the axis lines, font size, font
h = legend( p, s);
set(h,'FontSize',24);

figure;
p = {};
s = {};
for l=1:L
    pp = histogram(reshape(samples(l,:,10),1,num_samples),500,...
    'FaceColor',cc(l,:),'FaceAlpha',1,'EdgeColor',cc(l,:),...
    'Normalization','probability');
    hold on
    p = [p,pp];
    s = [s,['Layer  ' num2str(l-1)]];
end
grid on
xlabel('Delay','fontsize',30);
ylabel('Empirical distribution','fontsize',30);
xlim([1,40])
set(h,'location','northeast');
set(gca,'fontsize',24,'fontname','Times New Roman') % Sets the width of the axis lines, font size, font
h = legend( p, s);
set(h,'FontSize',24);

