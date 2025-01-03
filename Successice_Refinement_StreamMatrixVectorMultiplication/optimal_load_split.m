% Project: Streaming Iterative distributed computing
% Author: Homa Esfahanizadeh, Alejandro Cohen, Muriel Médard
% Last modified: 2022/07/14
% Goal: Optimal split for iterative jobs

function [kappa_vec,theta] = optimal_load_split ( gamma, c_vec, m_vec , sigma_vec , K , Omega )
    theta_min=0;
    theta_max=10^8;
    while ((theta_max - theta_min)>0.00000001)
        theta = 0.5*(theta_max + theta_min);
        a_vec = c_vec + (gamma * c_vec.^2);
        b_vec = (2*gamma*c_vec.*m_vec)+m_vec+(gamma*sigma_vec.^2);
        kappa_vec = b_vec./(2*gamma*(m_vec.^2)).*...
            (-1+sqrt(1+((4*gamma*(m_vec.^2).*max(theta-a_vec,0))./(b_vec.^2))));
        if (sum(kappa_vec) < (K*Omega))
            theta_min = theta;
        elseif (sum(kappa_vec) > (K*Omega))
        	theta_max = theta;
        end
    end      
end

