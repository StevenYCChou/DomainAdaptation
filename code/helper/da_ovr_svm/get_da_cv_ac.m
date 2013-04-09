function [ac] = get_da_cv_ac(y_source, X_source, y_target, X_target, param, nr_fold)
%
%original framework: get_cv_ac.m downloaded from libSVM FAQ page
%http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/ovr_multiclass/get_cv_ac.m
%modified by Yen-Cheng Chou for domain adaptation
%
len=length(y_target);
ac = 0;
rand_ind = randperm(len);
for i=1:nr_fold % Cross training : folding
  test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
  train_ind = [1:len]';
  train_ind(test_ind) = [];
  model = ovrtrain([y_source; y_target(train_ind)], [X_source; X_target(train_ind,:)], param);
  [pred,a,decv] = ovrpredict(y_target(test_ind),X_target(test_ind,:),model);
  ac = ac + sum(y_target(test_ind)==pred);
end
ac = ac / len;
fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);
