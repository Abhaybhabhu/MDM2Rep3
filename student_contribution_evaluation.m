% Read data from the Excel file
data = ModelComparison;
data_oct_aug = table2array(data(:, 1:2));
data_oct_aug = str2double(data_oct_aug)
oct = data_oct_aug(:,1);
aug = data_oct_aug(:,2);

%%
% Plot the data
figure;
plot(oct); hold; plot(aug, '-r');
xlabel('time');
ylabel('prediction');
title('model predictions');
legend('October prediction', 'August prediction')

%%
pred_diff_raw = (oct - aug) ./ oct;

% replaces infinites with 0 
idxs = isinf(pred_diff_raw);
pred_diff_raw(idxs) = 0;

% Replace NaN values with 0
pred_diff_raw(isnan(pred_diff_raw)) = 0;

% replace negative values and 100%'s with 0 (100% occurs if 0 pred in aug)
for i = 1:length(pred_diff_raw)
    if pred_diff_raw(i) < 0
        pred_diff_raw(i) = 0;
    elseif pred_diff_raw(i) == 1
        pred_diff_raw(i) = 0;
    end
end

pred = pred_diff_raw*100;

concarr = [oct, aug, pred]
%% showing why should only consider latter two weeks
figure;
plot(pred);
xlabel('time');
ylabel('Student contribution percentage');
title('Plot of percentage student contribution');

%%
% Define start time
startTime = datetime('05-Oct-2023 01:00:00');

% Define end time
endTime = datetime('29-Oct-2023 19:45:00');

% Create a time series with 15-minute intervals
timeIntervals = startTime:minutes(15):endTime;

% Generate dates for each day in October
dates = datetime('05-Oct-2023'):datetime('29-Oct-2023');

figure;
plot(timeIntervals, pred);
xlabel('Date');
ylabel('Student contribution percentage');
title('Plot of percentage student contribution');
datetick('x', 'dd', 'keepticks');
xticks(dates);
xtickformat('dd-MMM');
xtickangle(45);


%% plot the two weeks 

first_week = pred(1026:1526);
second_week = pred(1698:2198);

figure;
plot(first_week); hold; plot(second_week, '-r');
xlabel('time');xlim([0, 500]);
ylabel('Student contribution percentage');
title('Overlaid plot of week 3 and 4');
legend('Week 3', 'Week 4');

%% add them up

combo = (first_week + second_week)/2;
figure;
plot(combo);xlim([0, 500]);
xlabel('time');ylabel('student contribution percentage');
title('plot of average student contribution a week')
%% daily average 
comb = zeros(96, 1);
for i = 1:5
    comb = comb + combo((i-1)*96+1:(i*96));
end
comb = comb/5;

% Define start time
startTime = datetime('04:45:00');

% Define end time
endTime = datetime('04:45:00') + hours(24);

% Create a linspace of time with 15-minute intervals
timeIntervals = startTime:minutes(15):endTime;

% Plot
figure;
plot(timeIntervals(1:end-1), comb(1:96),'-r', 'Linewidth', 2);
xlim([startTime, endTime]);
datetick('x', 'HH:MM', 'keepticks');
xlabel('Time');
ylabel('Percentage student contribution');
title('Average student contribution to pedestrian traffic for average term-time day');
