%input: matrix M
%similarity function  d(s1,s2)
%Smooth Kernel K_h(s1,s2)

[a, b] = get_anchor(M);
weighted_rows_coef = similarity_row(M,a);
weighted_cols_coef = similarity_col(M,b);
filtered_row_coefs = Epanechnikov_kernel_row;
filtered_col_coefs = Epanechnikov_kernel_col;
[U,V] = factorization(M, filtered_row_coefs, filtered_col_coefs);

function [a, b] = get_anchor(M)
    a = random('unif', 1, size(M,1));
    b = random('unif', 2, size(M,2));
end

function weighted_rows_coef = similarity_row (M,a)
    M_normalized = M ./ repmat( sqrt( sum(M.^2, 2) ) ,1 , size(M,2));
    row = M_normalized(a,:);
    weighted_coef_rows = row * M_normalized';  
end

function weighted_cols_coef = similarity_col (M,b)
    M_normalized = M ./ repmat( sqrt( sum(M.^2, 1) ) , size(M,1), 1);
    col = M_normalized(:,b);
    weighted_coef_cols = M_normalized' * col;  
end

function filtered_row_coefs = Epanechnikov_kernel_row (coefs, bandwidth)
    filtered_row_coefs = (ones(size(coefs))-coefs.^2) .* (coefs < bandwidth);
end

function filtered_col_coefs = Epanechnikov_kernel_col (coefs, bandwidth)
    filtered_col_coefs = (ones(size(coefs))-coefs.^2) .* (coefs < bandwidth);
end

