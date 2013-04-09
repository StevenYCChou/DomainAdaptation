%Author: Yen-Cheng Chou
%Date: 27th Aug., 2012
%
load ../dat/ar_subset.mat
classNum = 20;


%each vector 
nPixels = size(Xo,1);
meanAcc = zeros(1,classNum);

dataSize = size(Xo,2);
groupSize = floor(dataSize/3);

XoLabel = kron(1:20,ones(1,9))';
[XoPermute, XoLabelPermute, idxPermute] = dataPerm(Xo,XoLabel);

%idx_1 = idxPermute(            1:   groupSize,1);
%idx_2 = idxPermute(  groupSize+1: 2*groupSize,1);
%idx_3 = idxPermute(2*groupSize+1:    dataSize,1);
%[Xo_label_1 Xo_label_2 Xo_label_3 Xo_1 Xo_2 Xo_3] = splitThreeSet(XoPermute, XoLabelPermute);

accuracy = zeros(1,7);

trainingSet = cell(1,3);
trainingSet_label = cell(1,3);
model = cell(1,3);
W = cell(20,3);
B = zeros(20,3);


[data_1 data_2 data_3] = splitThreeSet(XoPermute);
trainingSet{1} = [data_2 data_3];
trainingSet{2} = [data_1 data_3];
trainingSet{3} = [data_1 data_2];

for log10c = -3:3
    for nClass = 1:classNum

        binaryLabel = -1 * ones(180,1);
        binaryLabel(XoLabelPermute==nClass) = 1;

        [label_1 label_2 label_3] = splitThreeSetVert(binaryLabel);
        
        %%

        trainingSet_label{1} = vertcat(label_2, label_3);
        trainingSet_label{2} = vertcat(label_1, label_3);
        trainingSet_label{3} = vertcat(label_1, label_2);
        
        model{1} = svmtrain(trainingSet_label{1}, trainingSet{1}', ['-s 0 -t 0 -c ', num2str(10^log10c)]);
        model{2} = svmtrain(trainingSet_label{2}, trainingSet{2}', ['-s 0 -t 0 -c ', num2str(10^log10c)]);
        model{3} = svmtrain(trainingSet_label{3}, trainingSet{3}', ['-s 0 -t 0 -c ', num2str(10^log10c)]);
        %%
        %extract out w and b
        for nModel = 1:3
            w =  model{nModel}.SVs' * model{nModel}.sv_coef;
            b = -model{nModel}.rho;
            if (model{nModel}.Label(1) == -1)
                w = -w;
                b = -b;
            end
            W{nClass,nModel} = w;
            B(nClass,nModel) = b;
        end
             %%   
        %trainingSet
        
        
        %acc = zeros(3);
        %[junk1, acc(:,1), junk2] = svmpredict(label_1, data_1', model_1);
        %[junk1, acc(:,2), junk2] = svmpredict(label_2, data_2', model_2);
        %[junk1, acc(:,3), junk2] = svmpredict(label_3, data_3', model_3);
        %meanAcc(1,i) = mean(acc(1,:));

    end
    
    
    [v1 idx1] = max(cell2mat(W(:,1)')' * data_1 + repmat(B(:,1),1,60),[],1);
    [v2 idx2] = max(cell2mat(W(:,2)')' * data_2 + repmat(B(:,2),1,60),[],1);
    [v3 idx3] = max(cell2mat(W(:,3)')' * data_3 + repmat(B(:,3),1,60),[],1);
    
    labelEst(1:60,1) =  (idx1.*(v1>0))';
    labelEst(61:120,1) =  (idx2.*(v2>0))';
    labelEst(121:180,1) =  (idx3.*(v3>0))';
    
    isCorrect = (XoLabelPermute == labelEst);
    
    accuracy(log10c+4) = mean(isCorrect);
end

h = figure('Position', [20 20 500 500]);
bar(-3:3,accuracy);
xlabel('log_2(C)');
ylabel('accuray rate');

saveas(h, '../figure/Ctest.jpg');
close all;