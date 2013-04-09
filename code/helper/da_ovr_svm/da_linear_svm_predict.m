function [acc, predict] = da_linear_svm_predict( ...
    X_source_train, y_source_train, ...
    X_target_train, y_target_train, ...
    X_target_test, y_target_test,   ...
    c_list, n_fold)   

% Jan. 2013
% This matlab code find the best parameter {Cost(c) and Gamma(g)} of 
% RBF-svm for domain adaptation via cross validation.
%% I/O description
% INPUT(required):
% X_source_train = d * n_{s,train} matrix of source domain training data
% X_target_train = d * n_{t,train} matrix of target domain training data
% X_target_test = d * n_{t,test} matrix of target domain testing data
% y_source = n_{s,train} * 1 array. label of source domain training data
% y_target = n_{t,train} * 1 array. label of target domain training data
% y_target = n_{t,test} * 1 array. label of target domain testing data
%
% INPUT(optional):
% c_list = 1 * n_c array, which is the grid of cost
% g_list = 1 * n_g array, which is the grid of gamma
% n_fold is arbitrary positive integer which means n fold cross validation
%
% OUTPUT:
% acc = accuracy with the best parameter
%
%% Copyright Information
% Yen-Cheng Chou, Jan. 2013.
% If you have any question, please email to "yencheng.chou@gmail.com".
% Copyright: Multimedia and Machine Learning Laboratory,
%            Research Center for Information Technology,
%            Academia Sinica, Taipei

if nargin < 7
    c_list = 2.^([-10:2:20]);
end
if nargin < 8
    n_fold = 3;
end

[bestc, bestcv] = get_da_svm_cv_best_parameter(X_source_train, y_source_train, X_target_train, y_target_train, c_list, n_fold);

X_source_train = X_source_train';
X_target_train = X_target_train';
X_target_test  = X_target_test' ;

cmdbest = ['-q -c ', num2str(bestc)];
model = ovrtrain([y_source_train; y_target_train], [X_source_train; X_target_train], cmdbest);
[predict acc decv] = ovrpredict(y_target_test, X_target_test, model);
fprintf('Accuracy = %g%%\n', acc * 100);

end

