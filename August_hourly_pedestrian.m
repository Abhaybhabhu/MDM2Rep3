filename = '2023-08-01_2023-08-31_counts.csv'; 
opts = detectImportOptions(filename);
opts = setvartype(opts, 'Pedestrian', 'double'); 
data = readtable(filename, opts);

data.UTCDatetime = datetime(data.UTCDatetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

data.DayOfWeek = weekday(data.UTCDatetime); 
data.HourOfDay = hour(data.UTCDatetime);

weeklyHourlyCounts = varfun(@sum, data, 'InputVariables', 'Pedestrian','GroupingVariables', {'DayOfWeek', 'HourOfDay'});

daysInWeek = [4; 4; 5; 5; 5; 4; 4];
for i = 1:height(weeklyHourlyCounts)
    weeklyHourlyCounts.avg_Pedestrian(i) = weeklyHourlyCounts.sum_Pedestrian(i) / daysInWeek(weeklyHourlyCounts.DayOfWeek(i));
end

dayNames = {'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'};
colors = lines(7);
figure;
hold on;
for i = 1:7
    dayData = weeklyHourlyCounts(weeklyHourlyCounts.DayOfWeek == i, :);
    plot(dayData.HourOfDay, dayData.avg_Pedestrian, 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', dayNames{i});
end
legend('Location', 'northeastoutside');
xlabel('Hour of Day');
ylabel('Average Number of Pedestrians');
title('Average Hourly Pedestrian Counts by Day of Week for August 2023');
grid on;
hold off;