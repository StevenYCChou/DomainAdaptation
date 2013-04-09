function [model] = ovrtrain(y, x, cmd)
%downloaded from libSVM FAQ page
%http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/ovr_multiclass/ovrtrain.m

labelSet = unique(y);
labelSetSize = length(labelSet);
models = cell(labelSetSize,1);

for i=1:labelSetSize
    models{i} = svmtrain(double(y == labelSet(i)), x, cmd);
end

model = struct('models', {models}, 'labelSet', labelSet);

end
