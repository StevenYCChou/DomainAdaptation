[trainY trainX] = libsvmread('../dat/dna.scale');
[testY testX] = libsvmread('../dat/dna.scale.t');
model = ovrtrain(trainY, trainX, '-c 8 -g 4');
[pred ac decv] = ovrpredict(testY, testX, model);
fprintf('Accuracy = %g%%\n', ac * 100);
%Conduct CV on a grid of parameters
bestcv = 0;
for log2c = -1:2:3,
  for log2g = -4:2:1,
    cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = get_cv_ac(trainY, trainX, cmd, 3);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end