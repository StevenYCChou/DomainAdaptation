function W = DDA (D_source, D_target, y_source, y_target, lambda_source, lambda_target)
% Jan. 2013
% This matlab code implements the Discriminative Domain Adaptation.
%% I/O description
% INPUT:
% D_source = d * n_s matrix of source domain data
% D_target = d * n_t matrix of target domain data
% y_source = n_s * 1 array. label of source domain data
% y_target = n_t * 1 array. label of target domain data
% lambda_source = parameter for solving Principal Component Pursuit (PCP)
%                 on source domain data
% lambda_target = parameter for solving Principal Component Pursuit (PCP)
%                 on target domain data
% OUTPUT:
% W = d * d matrix, which is linear transformation from source domain to
%     target domain.
%% Algorithm Description
% Step1: Obtain the hidden structure A_source, A_target by Pricipal
%        Component Pursuit
% Step2: Obtain the closed form W by solving the optimization:
%              max_W sum_{i,j} [ (W*A_source)' * A_target ] .* L
%              s.t.  WW' = I
%        W can be obtained by closed form V*U',
%        where [U,S,V] = svd(A_source * L * A_target')
%% Copyright Information
% Yen-Cheng Chou, Jan. 2013.
% If you have any question, please email to "yencheng.chou@gmail.com".
% Copyright: Multimedia and Machine Learning Laboratory,
%            Research Center for Information Technology,
%            Academia Sinica, Taipei
%% Reference
% Reference: [1] E.J. Candes, X. Li, Y. Ma, and J. Wright, "Robust
%                principal component analysis?", JACM, 2011.
%                inexact_alm_rpca MATLAB code can be downloaded from
%                http://perception.csl.illinois.edu/matrix-rank/sample_code.html#RPCA
%                which is copyright by:
%                  Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%                  Microsoft Research Asia, Beijing
%            [2] R.A. Horn and C.R. Johnson, "Matrix Analysis," ch. 7.4,
%                p.432, Cambridge University Press, 1990

addpath ../code/helper/inexact_alm_rpca ../code/helper/inexact_alm_rpca/PROPACK/

%% Step 1: Obtain the hidden structures.
%%% Obtain hidden Structures A_source from D_source, and A_target from D_target
%%% by Principal Component Pursuit algorithm(equivalent to the RPCA algorithm)

[A_source, E_source] = inexact_alm_rpca(D_source, lambda_source);
[A_target, E_target] = inexact_alm_rpca(D_target, lambda_target);

%%% get the label consistency matrix L
L = build_label_consistency(y_source, y_target);

%% Step 2: Obtain the closed form W
[Uf Sf Vf] = svd( A_source * L * A_target' ,'econ');
W = Vf*Uf';

end

function L = build_label_consistency(y_source, y_target)
L = zeros(length(y_source),length(y_target));
for i = 1:length(y_source)
    for j = 1:length(y_target)
        if y_source(i) == y_target(j)
            L(i,j) = 1;
        else
            L(i,j) = 0;
        end
    end
end
end