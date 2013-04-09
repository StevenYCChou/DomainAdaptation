function [bestc, bestcv] = get_da_linear_svm_cv_best_parameter(X_source, y_source, X_target, y_target, c_list, n_fold)
% Jan. 2013
% This matlab code find the best parameter {Cost(c) and Gamma(g)} of 
% RBF-svm for domain adaptation via cross validation.
%% I/O description
% INPUT(required):
% X_source = d * n_s matrix of source domain data
% X_target = d * n_t matrix of target domain data
% y_source = n_s * 1 array. label of source domain data
% y_target = n_t * 1 array. label of target domain data
%
% INPUT(optional):
% c_list = 1 * n_c array, which is the grid of cost
% g_list = 1 * n_g array, which is the grid of gamma
% n_fold is arbitrary positive integer which means n fold cross validation
% OUTPUT:
% bestc = best Cost
% bestg = best Gamma
% bestcv = under the parameter bestc and bestg, the best cross validation
%          accuracy it achieve.
%
%% Copyright Information
% Yen-Cheng Chou, Jan. 2013.
% If you have any question, please email to "yencheng.chou@gmail.com".
% Copyright: Multimedia and Machine Learning Laboratory,
%            Research Center for Information Technology,
%            Academia Sinica, Taipei

if nargin < 5
    c_list = 2.^([-5:2:3]);
end
if nargin < 6
    n_fold = 3;
end

%% TRANSPOSE data for libSVM
    X_source = X_source';
    X_target = X_target';
    
    %% ONE-VS-REST SVM
    %Conduct CV on a grid of parameters
    bestcv = 0; bestc = 0; 
    for c = c_list
      for g = g_list
        cmd = ['-q -c ', num2str(c)];
        cv = get_da_cv_ac(y_source, X_source, y_target, X_target, cmd, n_fold);
        if (cv >= bestcv),
          bestcv = cv; bestc = c;
        end
        fprintf('%g %g (best c=%g, rate=%g)\n', c, cv, bestc, bestcv);
      end
    end

end

