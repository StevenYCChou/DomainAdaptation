%Author: Yen-Cheng Chou
%Date: 27th Aug., 2012
%

load ../dat/ar_subset.mat
classNum = 20;

%each vector 
nPixels = size(Xo,1);
nFaces = size(Yo,2);
meanAcc = zeros(1,classNum);

dataSize = size(Xo,2);
groupSize = floor(dataSize/3);

XoLabel = kron(1:20,ones(1,9))';
YoLabel = kron(1:20,ones(1,9))';

W = zeros(nPixels,classNum);
B = zeros(classNum,1);

for nClass = 1:classNum

    binaryLabel = -1 * ones(180,1);
    binaryLabel(XoLabel==nClass) = 1;

    model = svmtrain(binaryLabel, Xo', '-s 0 -t 0 -c 1');

    w =  model.SVs' * model.sv_coef;
    b = -model.rho;
    if (model.Label(1) == -1)
        w = -w;
        b = -b;    
    end
    
    W(:,nClass) = w;
    B(nClass,1) = b;
    
end

[value idx] = max(W' * Yo + repmat(B,1,nFaces), [], 1);
labelEst =  (idx.*(value>0))';
isCorrect = (YoLabel == labelEst);
accuracyMy = mean(isCorrect);

 modelLib = svmtrain(XoLabel,Xo', '-s 0 -t 0 -c 1');
[junk accuracyLib junk] = svmpredict(YoLabel,Yo', modelLib);