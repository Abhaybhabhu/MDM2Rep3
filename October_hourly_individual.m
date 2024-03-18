filename = '2023-10-01_2023-10-31_counts.csv';
opts = detectImportOptions(filename);
opts = setvartype(opts, 'Pedestrian', 'double'); 
data = readtable(filename, opts);

data.UTCDatetime = datetime(data.UTCDatetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

data.DayOfWeek = weekday(data.UTCDatetime); 
data.HourOfDay = hour(data.UTCDatetime);

weeklyHourlyCounts = varfun(@sum, data, 'InputVariables', 'Pedestrian', 'GroupingVariables', {'DayOfWeek', 'HourOfDay'});

daysInWeek = [5; 5; 5; 4; 4; 4; 4]; 
for i = 1:height(weeklyHourlyCounts)
    weeklyHourlyCounts.avg_Pedestrian(i) = weeklyHourlyCounts.sum_Pedestrian(i) / daysInWeek(weeklyHourlyCounts.DayOfWeek(i));
end

dayNames = {'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'};
colors = lines(7); 

for dayIndex = 1:7
    figure;
    dayData = weeklyHourlyCounts(weeklyHourlyCounts.DayOfWeek == dayIndex, :);
    plot(dayData.HourOfDay, dayData.avg_Pedestrian, 'Color', colors(dayIndex, :), 'LineWidth', 2);
    xlabel('Hour of Day');
    ylabel('Average Number of Pedestrians');
    title(['Average Hourly Pedestrian Counts for ', dayNames{dayIndex}, ' in August 2023']);
    grid on;
end