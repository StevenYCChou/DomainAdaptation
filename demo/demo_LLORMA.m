addpath '../code/main_algorithm/LLORMA/'
addpath '../code/helper/'
M = [ 1  2  3  4  5;
      1  2  3  4  5;
      2  4  6  8 10;
      1  2  3  0  5;
      5  4  3  2  1;
      0  0  0  0  3]
  
[U,V,a,b] = LLORMA(M);
a
b
local_low_rank_matrix = U*V;