fp = fopen('bench_wrt_popsize.txt');
data = textscan(fp, '%12n%12n%12n%12n%12n%12n', ...
  'HeaderLines', 1, 'ReturnOnError', 0);
fclose(fp);
data = cell2mat(data);

subplot(1,2,1);
loglog(data(:,1), data(:,3) * 1e-3, '^-', ...
       data(:,1), data(:,5) * 1e-3, '^--', ...
       data(:,1), data(:,6) * 1e-3, 's--');
title({'One Generation with', '10000 Training Instances'}, ...
  'FontSize',12,'FontWeight','b');
legend('x86', 'GTX260', 'GTX275', 'Location', 'NorthWest');
xlabel('Population size (number of ANNs)');
ylabel('Execution time (s)');

fp = fopen('bench_wrt_trainsize.txt');
data = textscan(fp, '%12n%12n%12n%12n%12n%12n', ...
  'HeaderLines', 1, 'ReturnOnError', 0);
fclose(fp);
data = cell2mat(data);

subplot(1,2,2);
loglog(data(:,2), data(:,3) * 1e-3, '^-', ...
       data(:,2), data(:,5) * 1e-3, '^--', ...
       data(:,2), data(:,6) * 1e-3, 's--');
title({'One Generation of', '100 ANNs'}, ...
  'FontSize', 12, 'FontWeight', 'b');
legend('x86', 'GTX260', 'GTX275', 'Location', 'NorthWest');
xlabel('Training set size (number of instances)');
%ylabel('Execution time (s)');
