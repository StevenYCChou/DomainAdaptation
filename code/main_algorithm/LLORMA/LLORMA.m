%input: matrix M
%similarity function  d(s1,s2)
%Smooth Kernel K_h(s1,s2)

function [U,V,a,b] = LLORMA(M)
bw = 0.3;

 
EPSILON = 1e-6;

[a, b] = get_anchor(M);
distance_rows = get_row_distance(M,a);
distance_cols = get_col_distance(M,b);

%distance_rows = get_row_distance(M,1);
%distance_cols = get_col_distance(M,1);
filtered_row_coefs = Epanechnikov_kernel_row(distance_rows, bw);
filtered_col_coefs = Epanechnikov_kernel_col(distance_cols, bw);

local_matrix = repmat(filtered_row_coefs,1, size(M,2)) .* repmat(filtered_col_coefs, size(M,1),1) .* M;
local_matrix(local_matrix==0) = EPSILON;
[U,V] = nmf(local_matrix, 2, 0);
%[U,V] = factorization(M, filtered_row_coefs, filtered_col_coefs);
end

function [a, b] = get_anchor(M)
    a = randperm(size(M,1));
    a = a(1);
    b = randperm(size(M,2));
    b = b(1);
end

function distances = get_row_distance (M,a)
    M_normalized = M ./ repmat( sqrt( sum(M.^2, 2) ) ,1 , size(M,2));
    row = M_normalized(a,:);
    distances = ones(size(M,1),1) - M_normalized * row';  
end

function distances = get_col_distance (M,b)
    M_normalized = M ./ repmat( sqrt( sum(M.^2, 1) ) , size(M,1), 1);
    col = M_normalized(:,b);
    distances = ones(1,size(M,2)) - col' * M_normalized;  
end

function filtered_row_coefs = Epanechnikov_kernel_row (row_distances, bandwidth)
    filtered_row_coefs = (1-row_distances.^2) .* (row_distances < bandwidth);
end

function filtered_col_coefs = Epanechnikov_kernel_col (col_distances, bandwidth)
    filtered_col_coefs = (1-col_distances.^2) .* (col_distances < bandwidth);
end

