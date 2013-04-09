function [acc_best, var_best, lambda_source_best, lambda_target_best] = demo_DDA( domain_source, domain_target, lambda_source_list, lambda_target_list)

addpath ../code/helper/svm ../code/helper/da_ovr_svm    % svm method
addpath ../code/main_algorithm/DDA                       % our method

[num_training_A, num_training_B, XA, yA, XB, yB] = load_domain_adaptation_data(domain_source,domain_target);

numclasses = length(unique(yA));

RUNS_MIN = 1;
RUNS_MAX = 20;

acc_best = 0;
var_best = 0;
lambda_source_best = 0;
lambda_target_best = 0;

for ls = 1:length(lambda_source_list)
    lambda_source = lambda_source_list(ls);
    for lt = 1:length(lambda_target_list)
        lambda_target = lambda_target_list(lt);
        
        for ITER = RUNS_MIN:RUNS_MAX
            disp(sprintf('Run %d',ITER));
            
            training_indA=[]; training_indB=[]; testing_indB=[];
            
            for i = 1:numclasses
                %%%%%%% domain A %%%%%%%
                indA = find(yA==i);
                rpA=randperm(length(indA));
                training_indA = [training_indA indA(rpA(1:num_training_A))];
                %%%%%%% domain B %%%%%%%
                indB = find(yB==i);
                rpB=randperm(length(indB));
                training_indB = [training_indB indB(rpB(1:num_training_B))];
                testing_indB = [testing_indB indB(rpB(num_training_B+1:end))];
            end
            
            XA_training = XA(:,training_indA)  ;
            yA_training = yA(training_indA)'   ;
            
            XB_training = XB(:,training_indB)  ;
            yB_training = yB(training_indB)'   ;
            
            XB_testing = XB(:,testing_indB)    ;
            yB_testing = yB(testing_indB)'     ;
            
            %% Our Method: DDA
            W = DDA (XA_training, XB_training, yA_training, yB_training, lambda_source, lambda_target);
            XA_training_adapted = W * XA_training;
            
            XAB_training = [XA_training_adapted, XB_training];
            yAB_training = [yA_training; yB_training];
            
            [acc(ITER), predict] = da_svm_predict( XA_training_adapted, yA_training, XB_training, yB_training, XB_testing, yB_testing) ;
            
        end
        
        mean_acc = mean(acc);
        
        if(mean_acc > acc_best)
            acc_best = mean_acc;
            var_best = var(acc);
            lambda_source_best = lambda_source;
            lambda_target_best = lambda_target;
        end
    end
    
end
%% Save data
eval(['save ../savedata/exp_DDA_OC10_',domain_source, '_to_', domain_target, '_' ,sprintf('%d.%d.%d.%02d-%02d-%02d', fix(clock)) ,'.mat']);

end

function [num_training_A, num_training_B, XA, yA, XB, yB] = load_domain_adaptation_data(source_name,target_name)

source_domain_name = get_domain (source_name); 
target_domain_name = get_domain (target_name);

domain = {'amazon','Caltech10','webcam','dslr'};
switch lower(source_name)
    case {'a','c'}
        num_training_A = 20;
    case {'d', 'w'}
        num_training_A = 8;
    otherwise
        error('unknown domain.');
end
num_training_B = 3;

load(['../dataset/OC10/',source_domain_name,'_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));

XA = fts';
yA = labels';

load(['../dataset/OC10/',target_domain_name,'_SURF_L10.mat']);
fts = fts ./ repmat(sum(fts,2),1,size(fts,2));

XB = fts';
yB = labels';

end

function [d] = get_domain (name) 
    domain = {'amazon','Caltech10','webcam','dslr'};
    switch lower(name) 
        case 'a'
            d = domain{1};
        case 'c'
            d = domain{2};
        case 'w'
            d = domain{3};
        case 'd'
            d = domain{4};
        otherwise
            error('unknown domain.');
    end
end