function [X,y] = load_data(abbrev)

prefix = '../database/OC10/';
postfix = '_SURF_L10.mat';

switch lower(abbrev)
    case 'a'
        domain_name = 'amazon';
    case 'c'
        domain_name = 'Caltech10';
    case 'w'
        domain_name = 'webcam';
    case 'd'
        domain_name = 'dslr';
    otherwise
        error('unknown domain.');
end

load([prefix,domain_name,postfix]);
X = fts ./ repmat(sum(fts,2),1,size(fts,2));
X = X';
y = labels';

end