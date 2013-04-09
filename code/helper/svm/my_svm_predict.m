function [acc, predict] = my_svm_predict( ...
    X_train, y_train, ...
    X_test, y_test,   ...
    c_list, g_list, n_fold)  
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
    c_list = 2.^([-5:2:3]);
end
if nargin < 8
    g_list = 2.^([-5:2:3]);
end
if nargin < 9
    n_fold = 3;
end

[bestc, bestg, bestcv] = get_my_svm_cv_best_parameter(X_train, y_train);

X_train = X_train';
X_test  = X_test' ;

cmdbest = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = ovrtrain([y_train; y_train], [X_train; X_train], cmdbest);
[predict acc decv] = ovrpredict(y_test, X_test, model);
fprintf('Accuracy = %g%%\n', acc * 100);

end

