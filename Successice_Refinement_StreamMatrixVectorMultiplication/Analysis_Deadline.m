% Project: Streaming Iterative distributed computing
% Author: Homa Esfahanizadeh, Alejandro Cohen, Muriel Médard
% Last modified: 2022/07/29
% Goal: Sucess rate versus deadline for layered-resolution computations

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
Omega = 1.2; % redundancy ratio
gamma = 1; % relative importance of the first moment and the second moment
P = 5; % number of workers
rng(900) % a random seed -- to realize the same workers and job arrivals 
interarrival_time = exprnd(1/lambda,1,J); % interrival time of the jobs, exponentially distributed
mu_vec = rand(1,P);
mu_vec = (Z*Omega*lambda) * (4*mu_vec/sum(mu_vec)); % computing rate
arrival_time_vec = cumsum(interarrival_time); % job arrival time

deadline_vec = 0:1:30;

failure = zeros (1,J,length(deadline_vec));
failure_layer = zeros (L,J,length(deadline_vec));

% no layering
C = Z / K; % computational complexity of each mega task per job
m_vec = C./ mu_vec;
sigma_vec = C./mu_vec;

for ind = 1:length(deadline_vec)
    deadline = deadline_vec(ind);
    disp(deadline);

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
        
        if ( j < J ) % not the last job
            if ( end_time_vec(j) > arrival_time_vec(j+1) ) % next job in the queue
                if ( job_progress(K) > deadline ) % deadline is reached
                    failure (1,j,ind) = 1;
                    end_time_vec(j) = arrival_time_vec(j+1);
                end
            end
        end
    end
end

% layering
J_mini = [1:m,m-1:-1,1]; % number of mini-jobs (matrix-matrix multiplications) per layer
C_mini = C / m / m; % computational complexity of each mini task
m_mini_vec = C_mini ./ mu_vec;
sigma_mini_vec = C_mini ./ mu_vec;

for ind = 1:length(deadline_vec)
    deadline = deadline_vec(ind);
    disp(deadline);
    
    start_comp_vec = zeros(1,J); % assignment processing start per job
    end_time_vec = zeros (J,L); % job layer end time (K tasks are finished)
    
    for j = 1:J
        if ( j==1 )
            start_comp_vec(j) = arrival_time_vec(j);
        else
            start_comp_vec(j) = max(end_time_vec(j-1,L),arrival_time_vec(j));
        end
        
        terminated = 0; % A flag that shows the job is not terminated yet
        
        for l = 1:L
            if ( terminated == 1 )
                failure_layer (l,j,ind) = 1;
                end_time_vec(j,l) = arrival_time_vec(j+1);
            else
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
                if ( j < J ) % not the last job
                    if ( end_time_vec(j,l) > arrival_time_vec(j+1) ) % next job in the queue
                        if ( (end_time_vec(j,l) - start_comp_vec(j) ) > deadline ) % deadline is reached
                            failure_layer (l,j,ind) = 1;
                            end_time_vec(j,l) = arrival_time_vec(j+1);
                            terminated = 1;
                        end
                    end
                end
            end
        end
    end
end

figure;
p = {};
s = {};
hold on
pp = plot(deadline_vec,1-sum(reshape(failure,J,length(deadline_vec)))/J ,'k-' , 'linewidth',2 );
p = [p,pp];
s = [s,'No layering'];
cc = hsv(L); % creates colormap
cc(2,:)=[0,0.5,0];
for l=1:L
    pp = plot(deadline_vec,1-sum(reshape(failure_layer(l,:,:),...
        J,length(deadline_vec)))/J , 'color',cc(l,:),'linewidth',2 );
    p = [p,pp];
    s = [s,['Layer  ' num2str(l-1)]];
end
xline(10,'--k','linewidth',2);
h = legend( p, s);
set(h,'FontSize',24);
set(gca,'fontsize',24,'fontname','Times New Roman') % Sets the width of the axis lines, font size, font

h.Position(1) = .5 - h.Position(3)/2;
h.Position(2) = .4 - h.Position(4)/2;
grid on
xlabel('Deadline','fontsize',30);
ylabel('Success rate','fontsize',30);

