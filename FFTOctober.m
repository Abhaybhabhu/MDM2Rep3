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
    title(['Average Hourly Pedestrian Counts for ', dayNames{dayIndex}, ' in October 2023']);
    grid on;
end

combinedHourlyCounts = varfun(@mean, weeklyHourlyCounts, 'InputVariables', 'avg_Pedestrian', 'GroupingVariables', 'HourOfDay');

y = combinedHourlyCounts.mean_avg_Pedestrian;
N = length(y);

Fs = 1; 

Y = fft(y);

P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(N/2))/N;

figure;
plot(f, P1);
title('Single-Sided Amplitude Spectrum of Combined Hourly Pedestrian Counts for October');
xlabel('Frequency (cycles per hour)');
ylabel('|P1(f)|');
grid on;

[~, maxIndex] = max(P1(2:end));
predominantFrequency = f(maxIndex + 1);
disp(['Predominant Frequency: ', num2str(predominantFrequency), ' cycles per hour']);
