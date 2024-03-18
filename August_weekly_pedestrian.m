filename = '2023-08-01_2023-08-31_counts.csv'; 
opts = detectImportOptions(filename);
opts = setvartype(opts, 'Pedestrian', 'double'); 
data = readtable(filename, opts);

data.UTCDatetime = datetime(data.UTCDatetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

data.DayOfWeek = weekday(data.UTCDatetime);

weeklyCounts = varfun(@sum, data, 'InputVariables', 'Pedestrian','GroupingVariables', 'DayOfWeek');

daysInWeek = [4; 4; 5; 5; 5; 4; 4];
weeklyCounts.avg_Pedestrian = weeklyCounts.sum_Pedestrian ./ daysInWeek;

dayNames = {'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'};
figure;
plot(weeklyCounts.DayOfWeek, weeklyCounts.avg_Pedestrian);
set(gca, 'xticklabel', dayNames);
xlabel('Day of Week');
ylabel('Average Number of Pedestrians');
title('Average Daily Pedestrian Counts by Day of Week for August 2023');
grid on;