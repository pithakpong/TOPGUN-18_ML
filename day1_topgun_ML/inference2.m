% Parameters
% Load the model
modelData = load('model.mat'); % Load the .mat file
model = modelData.model; % Extract the model variable from the loaded struct
inputFile = 'Day2/Day2.wav'; % Long audio file for testing
windowLength = 1; % Window length in seconds
overlapRatio = 0.9; % 90% overlap for detection resolution
eventThreshold = 0.5; % Confidence threshold for event classification
suppressTime = 2; % Time period in seconds to suppress duplicate detections
Fs = 48000; % Sampling rate of the audio file
nfft = 512; % FFT length for spectrogram
overlap = round(windowLength * Fs * overlapRatio); % Calculate overlap in samples
startThreshold = db2mag(-30); % Threshold for sync signal detection

% Load and normalize long audio file to range [-1, 1]
[x, Fs] = audioread(inputFile);
x = double(x(:)); % Ensure x is a column vector and of type double
x = x / max(abs(x)); % Normalize entire audio file to range [-1, 1]

% Detect sync signal
syncDetected = false;
syncSamplePosition = 1;

for sampleIdx = 1:length(x)
    % Check if sync signal is detected
    if x(sampleIdx) > startThreshold
        syncDetected = true;
        syncSamplePosition = sampleIdx + 1; % Start processing from the next sample after sync
        fprintf('Sync signal detected at sample %d. Starting processing at relative sample 1.\n', ...
                sampleIdx);
        break;
    end
end

% Start processing if sync signal is detected
if syncDetected
    % Load and normalize 0.1-second reference sounds for "N" and "F" events
    [refNormal, refFs] = audioread('Reference/Normal_1.wav'); 
    [refFaulty, ~] = audioread('Reference/Faulty_2.wav'); 

    % Trim reference signals to 0.1 seconds and normalize to [-1, 1]
    refNormal = refNormal(1:min(round(0.1 * refFs), length(refNormal)));
    refFaulty = refFaulty(1:min(round(0.1 * refFs), length(refFaulty)));
    refNormal = refNormal / max(abs(refNormal)); % Normalize
    refFaulty = refFaulty / max(abs(refFaulty)); % Normalize

    % Convert parameters to samples
    windowSamples = round(windowLength * Fs);
    stepSamples = windowSamples - overlap;

    % Initialize storage for detected events
    eventCounter = 1;
    eventTimes = [];
    samplePositions = [];
    eventLabels = [];
    lastEventTime = -suppressTime; % Initialize last event time far in the past

    % Initialize buffer for sliding window
    buffer = zeros(windowSamples, 1);
    relativeSample = 1; % Reset relative sample counter to start at 1 after sync

    % Sliding window loop, starting from syncSamplePosition
    currentSample = syncSamplePosition;
    while currentSample + stepSamples <= length(x)
        % Fill buffer with current window of data
        buffer(1:end-stepSamples) = buffer(stepSamples+1:end); % Shift buffer
        buffer(end-stepSamples+1:end) = x(currentSample:currentSample + stepSamples - 1); % Add new samples

        % Normalize buffer to range [-1, 1]
        maxVal = max(abs(buffer));
        if maxVal > 0
            buffer = buffer / maxVal; % Normalize
        end

        % Window start time in seconds relative to sync
        windowStartTime = relativeSample / Fs;

        % Feature extraction
        [S, F, T] = spectrogram(buffer, hamming(windowSamples), overlap, nfft, Fs);
        meanSpectrum = mean(abs(S), 2);
        spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
        spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
        rmsValue = sqrt(mean(buffer.^2));
        zeroCrossings = sum(abs(diff(sign(buffer)))) / (2 * length(buffer));
        coeffs = mfcc(buffer, Fs, 'NumCoeffs', 13);
        meanMFCC = mean(coeffs, 1);

        % Pack features into feature vector
        featureVector = [spectralCentroid; spectralSpread; rmsValue; zeroCrossings; meanMFCC'];
        if length(featureVector) > numFeatures
            featureVector = featureVector(1:numFeatures);
        else
            featureVector = [featureVector; zeros(numFeatures - length(featureVector), 1)];
        end

        % Classify the segment
        try
            [predictedLabel, score] = predict(model, featureVector');
        catch exception
            fprintf('Prediction error at segment starting %.2f seconds: %s\n', windowStartTime, exception.message);
            continue; % Skip this segment if there's an error
        end

        % Handle both single output and cell array output
        if iscell(predictedLabel) && ~isempty(predictedLabel)
            label = predictedLabel{1};
        elseif ischar(predictedLabel) || isstring(predictedLabel)
            label = char(predictedLabel);
        else
            fprintf('Unexpected format in predictedLabel at segment %.2f seconds.\n', windowStartTime);
            continue;
        end

        % Retrieve the maximum score
        maxScore = max(score);

        % Process events classified as "N" or "F" for refined positioning
        if (strcmp(label, 'N') || strcmp(label, 'F')) && maxScore > eventThreshold
            % Select appropriate reference based on classification
            refSound = strcmp(label, 'N') * refNormal + strcmp(label, 'F') * refFaulty;

            % Apply cross-correlation to find precise event position
            [c, lag] = xcorr(buffer(1:round(0.1 * Fs)), refSound);
            [~, lagIdx] = max(c);

            % Calculate exact sample position relative to sync
            accurateSamplePosition = relativeSample + lag(lagIdx);

            % Suppress duplicate events within the specified time window
            if (windowStartTime - lastEventTime) >= suppressTime
                % Record event details
                eventTimes(eventCounter) = windowStartTime;
                samplePositions(eventCounter) = accurateSamplePosition;
                eventLabels{eventCounter} = label;
                fprintf('Event detected at %.3f seconds (Relative Sample Position: %d), Label: %s\n', ...
                        windowStartTime, accurateSamplePosition, eventLabels{eventCounter});

                % Update last event time
                lastEventTime = windowStartTime;
                eventCounter = eventCounter + 1;
            end
        end

        % Move to the next segment based on step samples
        currentSample = currentSample + stepSamples;
        relativeSample = relativeSample + stepSamples; % Update relative sample position
    end

    % Display detected events with accurate positions
    eventTable = table(eventTimes', samplePositions', eventLabels', ...
                       'VariableNames', {'Time (s)', 'Relative Sample Position', 'Event'});
    disp(eventTable);

else
    fprintf('Sync signal not detected.\n');
end
