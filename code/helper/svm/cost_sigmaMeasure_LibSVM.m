%Author: Yen-Cheng Chou
%Date: 27th Aug., 2012
%

load ../dat/ar_subset.mat;

classNum = 20;

XoNormalized = (Xo - repmat(min(Xo,[],1),size(Xo,1),1))*spdiags(1./(max(Xo,[],1)-min(Xo,[],1))',0,size(Xo,2),size(Xo,2));

for log10c = -3:3;
    for log10g = -3:3;
        model = svmtrain(XoLabel, XoNormalized',['-s 0 -t 2 -v 3 -c ', num2str(10^log10c), ' -g ', num2str(10^log10g)]);
        log10c
        log10g
    end
end

 