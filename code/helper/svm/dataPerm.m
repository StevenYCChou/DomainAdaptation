function [dataPermute, labelPermute, idxPermute] = dataPerm(data,label)
dataSize = size(data,2);
idxPermute = randperm(dataSize)';
dataPermute = data(:,idxPermute);
labelPermute = label(idxPermute,1);
end