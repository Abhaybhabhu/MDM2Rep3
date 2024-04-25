function specturalAnalysis(fileName)
    % Load data from the provided file
    try
        data = readtable(fileName);
    catch
        error('Failed to read data from the file. Please check the file format and path.');
    end

    % Convert the first column into datetime objects
    timestamps = datetime(datenum(data{:,1}), 'ConvertFrom', 'datenum');

    % Extract relevant columns
    directions = data(:,4); % Assuming direction is in the 4th column
    counts = data(:,5); % Assuming counts are in the 5th column

    % Convert directions and counts to arrays
    directions_array = table2array(directions);
    counts_array = table2array(counts);

    % Aggregate the data
    unique_timestamps = unique(dateshift(timestamps, 'start', 'day')); % Aggregate by day
    net_change = zeros(size(unique_timestamps));

    for i = 1:numel(unique_timestamps)
        day_idx = dateshift(timestamps, 'start', 'day') == unique_timestamps(i);

        % Create logical indices for 'in' and 'out' directions
        in_directions_idx = strcmp(directions_array, 'in') & day_idx;
        out_directions_idx = strcmp(directions_array, 'out') & day_idx;

        % Extract counts for 'in' and 'out' directions
        in_counts = counts_array(in_directions_idx);
        out_counts = counts_array(out_directions_idx);

        % Compute net change
        net_change(i) = sum(in_counts) - sum(out_counts);
    end

    % Compute the Fourier Transform
    N = length(net_change);
    Y = fft(net_change, N*10);  % Compute FFT with increased resolution

    % Frequency domain
    T = 1; % Sampling period in days (since we're aggregating by day)
    Fs = 1/T; % Sampling frequency
    f = (0:(N*10-1))/N/T/10; % Frequency vector in days^-1 (increased resolution)

    % Plot the lower frequency component (daily patterns)
    figure;
    subplot(2,1,1);
    plot(f, 2/N * abs(Y)); % Plot the magnitude of the FFT result
    title('Single-Sided Amplitude Spectrum - Lower Frequency Component');
    xlabel('Frequency (days^{-1})');
    ylabel('|Y(f)|');
    xlim([0 0.1]); % Limit to lower frequencies (up to 0.1 days^-1)

    % Find peaks for lower frequency component
    [pks_lower, locs_lower] = findpeaks(2/N * abs(Y), f, 'MinPeakDistance', 0.01, 'SortStr', 'descend');
    hold on;
    plot(locs_lower, pks_lower, 'ro'); % Plot peaks on the spectrum

    % Plot the higher frequency component
    subplot(2,1,2);
    plot(f*24, 2/N * abs(Y)); % Plot the magnitude of the FFT result in hours^-1
    title('Single-Sided Amplitude Spectrum - Higher Frequency Component');
    xlabel('Frequency (hours^{-1})');
    ylabel('|Y(f)|');
    xlim([0 24]); % Limit to 24 hours^-1

    % Find peaks for higher frequency component
    [pks_higher, locs_higher] = findpeaks(2/N * abs(Y), f*24, 'MinPeakDistance', 0.1, 'SortStr', 'descend');
    hold on;
    plot(locs_higher, pks_higher, 'ro'); % Plot peaks on the spectrum

    % Display frequency peaks and corresponding power for both components
    disp('Frequency Peaks and Corresponding Power - Lower Frequency Component:');
    disp([locs_lower(:) pks_lower(:)]);
    disp('Frequency Peaks and Corresponding Power - Higher Frequency Component:');
    disp([locs_higher(:) pks_higher(:)]);
end
