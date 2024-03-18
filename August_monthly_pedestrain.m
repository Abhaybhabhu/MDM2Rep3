filename = '2023-08-01_2023-08-31_counts.csv';
opts = detectImportOptions(filename);
opts = setvartype(opts, 'Pedestrian', 'double'); 
data = readtable(filename, opts);

data.UTCDatetime = datetime(data.UTCDatetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

data.DayOfMonth = day(data.UTCDatetime);

dailyCounts = varfun(@sum, data, 'InputVariables', 'Pedestrian','GroupingVariables', 'DayOfMonth');

figure;
plot(dailyCounts.DayOfMonth, dailyCounts.sum_Pedestrian);
xlabel('Day of Month');
ylabel('Number of Pedestrians');
title('Daily Pedestrian Counts for August 2023');
grid on;