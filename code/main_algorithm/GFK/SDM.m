function [nDim, Sigma_alpha, Sigma_beta] = SDM(pcs_S, pcs_T, pcs_ST)

[D_S, d_S] = size(pcs_S);
[D_T, d_T] = size(pcs_T);

for n = 1:min(d_S, d_T)
    P_S = pcs_S(:,1:n);
    P_T = pcs_T(:,1:n);
    P_ST = pcs_ST(:,1:n);
    
    R_S = null(P_S');
    R_T = null(P_T');
    R_ST = null(P_ST');
    
    Sigma_alpha = svd(P_S' * P_ST);
    Sigma_beta = svd(P_T' * P_ST);
    
    0.5 * ( sin(acos(Sigma_alpha(n))) + sin(acos(Sigma_beta(n))) )
    
    if (0.5 * (sin(acos(Sigma_alpha(n))) + sin(acos(Sigma_beta(n))))) >= 1-1e-6
        break;
    end
    
end

nDim = n;

end