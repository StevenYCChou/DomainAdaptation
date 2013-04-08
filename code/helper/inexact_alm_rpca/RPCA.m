%Author: Yen-Cheng Chou
%Data:Aug. 30th, 2012

load ../dat/AR_subset.mat;

data = Xo(:,1:9);
%for n = 1:10 ;
%    lambda = 0.001*n;
%    [A_hat E_hat iexact_alm_rpcater] = inexact_alm_rpca(data,lambda);
%    rk(n) = rank(A_hat);
%end

%n = 1:10;

%plot(n,rk);
[A_hat E_hat iexact_alm_rpcater] = inexact_alm_rpca(data,0.003);
figure('Position', [20 20 200 200]);
for n = 1:9
    subplot(2,9,n); 
    a= reshape(A_hat(:,n),165,120);
    imshow(a, [min(min(a)), max(max(a))]); title(['A:face ', num2str(n)]);
end

for n = 1:9
    subplot(2,9,9+n); 
    a= reshape(E_hat(:,n),165,120);
    imshow(a, [min(min(a)), max(max(a))]); title(['E:face ', num2str(n)]);
end