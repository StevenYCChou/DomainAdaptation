function value = kernal_rbk_value(gamma,Coef,W,data, nGroup)
    classNum = size(Coef, 1);
    dataSize = size(data,2);
    value = zeros(classNum, size(data,2));
    
    %W_ngroup = cell2mat(W(:,nGroup)');
    %W_totalLength = size(W_ngroup,2);
    
    %dataRep = kron(data, ones(1,W_totalLength));
    nGroup
    for nClass = 1:classNum
        nClass
        c = Coef{nClass, nGroup};
        w = W{nClass, nGroup};
        for nData = 1:size(data,2)   
            abs = w - repmat(data(:,nData),1,size(w,2));
            value(nClass, nData) = exp(-gamma* sum((abs.^2),1) )*c;
        end
    end
end