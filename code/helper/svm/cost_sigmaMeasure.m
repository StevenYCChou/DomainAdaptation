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

XoNormalized = (Xo - repmat(min(Xo,[],1),size(Xo,1),1))*spdiags(1./(max(Xo,[],1)-min(Xo,[],1))',0,size(Xo,2),size(Xo,2));

[XoPermute, XoLabelPermute, idxPermute] = dataPerm(XoNormalized,XoLabel);

clear Xu Yo Yu Xo;
%accuracy = zeros(7,5);

trainingSet = cell(1,3);

[data_1 data_2 data_3] = splitThreeSet(XoPermute);
trainingSet{1} = [data_2 data_3];
trainingSet{2} = [data_1 data_3];
trainingSet{3} = [data_1 data_2];
for log10g = -2:2
    for log10c = -3:3

        %model= cell(1,3);
        Coef = cell(20,3);
        W = cell(20,3);
        B = zeros(20,3);
        trainingSet_label = cell(1,3);
        
        for nClass = 1:classNum

            binaryLabel = -1 * ones(180,1);
            binaryLabel(XoLabelPermute==nClass) = 1;
            [label_1 label_2 label_3] = splitThreeSetVert(binaryLabel);
            %%

            trainingSet_label{1} = vertcat(label_2, label_3);
            trainingSet_label{2} = vertcat(label_1, label_3);
            trainingSet_label{3} = vertcat(label_1, label_2);

            model{1} = svmtrain(trainingSet_label{1}, trainingSet{1}', ['-s 0 -t 2 -c ', num2str(10^log10c), ' -g ', num2str(10^(log10g))]);
            model{2} = svmtrain(trainingSet_label{2}, trainingSet{2}', ['-s 0 -t 2 -c ', num2str(10^log10c), ' -g ', num2str(10^(log10g))]);
            model{3} = svmtrain(trainingSet_label{3}, trainingSet{3}', ['-s 0 -t 2 -c ', num2str(10^log10c), ' -g ', num2str(10^(log10g))]);
            %%
            %extract out w and b
            for nModel = 1:3
                w =  model{nModel}.SVs' ;
                b = -model{nModel}.rho  ;
                if (model{nModel}.Label(1) == -1)
                    w = -w;
                    b = -b;
                end
                Coef{nClass,nModel} = model{nModel}.sv_coef;
                W{nClass,nModel} = w;
                B(nClass,nModel) = b;
            end
            clear binaryLabel label_1 label_2 label_3 trainingSet_label
        end
            clear model
        
        [v1 idx1] = max(kernal_rbk_value(10^(log10g), Coef, W, data_1, 1)+b,[],1);
        [v2 idx2] = max(kernal_rbk_value(10^(log10g), Coef, W, data_2, 2)+b,[],1);
        [v3 idx3] = max(kernal_rbk_value(10^(log10g), Coef, W, data_3, 3)+b,[],1);
        
        labelEst = zeros(180,1);
        labelEst(1:60,1) =  (idx1.*(v1>0))';
        labelEst(61:120,1) =  (idx2.*(v2>0))';
        labelEst(121:180,1) =  (idx3.*(v3>0))';

        isCorrect = (XoLabelPermute == labelEst);
        
        accuracy(log10c+4,log10g+3) = mean(isCorrect);
        clear B Coef W labelEst;
        
    end
end
h = figure('Position', [20 20 500 500]);
bar(-3:3,accuracy);
xlabel('log_2(C)');
ylabel('accuray rate');

saveas(h, '../figure/C_Sigma_test.jpg');
close all;