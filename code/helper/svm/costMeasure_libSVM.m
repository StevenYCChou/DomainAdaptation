%Author: Yen-Cheng Chou
%Date: 27th Aug., 2012
%

load ../dat/ar_subset.mat;

classNum = 20;

model=zeros(classNum, 7);
for j = -3:3;
    for i = 1:classNum
    Xo_label = -1 * ones(180,1);
    Xo_label(9*(i-1)+1:9*i,1) = 1; 
    cost = power(10,j);
    model(i,j+4) = svmtrain(Xo_label, Xo', ['-v 3 -c ', num2str(cost)]);
    end
end