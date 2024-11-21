% Parameters
modelData = load('model.mat'); % Load the pre-trained model
model = modelData.model; % Extract the model
inputFile = 'Final.wav'; % Long audio file for testing
windowLength = 1; % Window length in seconds
overlapRatio = 0.9; % 90% overlap for detection resolution
eventThreshold = 0.6; % Confidence threshold for event classification
suppressTime = 2; % Suppression period for duplicate detections
Fs = 48000; % Sampling rate of the audio file
nfft = 512; % FFT length for spectrogram
overlap = round(windowLength * Fs * overlapRatio); % Calculate overlap in samples
startThreshold = db2mag(-30); % Threshold for sync signal detection

% Load and normalize 0.1-second reference sounds for "N" and "F" events
[refNormal, refFs] = audioread('Reference/Normal_1.wav'); 
[refFaulty, ~] = audioread('Reference/Faulty_2.wav'); 

% Trim reference signals to 0.1 seconds and normalize to [-1, 1]
refNormal = refNormal(1:min(round(0.1 * refFs), length(refNormal)));
refFaulty = refFaulty(1:min(round(0.1 * refFs), length(refFaulty)));
refNormal = refNormal / max(abs(refNormal)); % Normalize
refFaulty = refFaulty / max(abs(refFaulty)); % Normalize

% Initialize audio reader for frame-by-frame processing
fileReader = dsp.AudioFileReader(inputFile, 'SamplesPerFrame', round((1 - overlapRatio) * windowLength * Fs), 'PlayCount', 1);

% Convert parameters to samples
windowSamples = round(windowLength * Fs);
stepSamples = windowSamples - overlap;

% Initialize storage for detected events
eventCounter = 1;
eventTimes = [];
samplePositions = [];
eventLabels = [];
lastEventTime = -suppressTime; % Initialize last event time far in the past
nonStampingCounter = 1; % Counter for non-stamping events

% Initialize buffer for sliding window
buffer = zeros(windowSamples, 1);
syncDetected = false;
totalSamplesRead = 0; % Track the total samples read from the file

% Frame-by-frame processing
while ~isDone(fileReader)
    % Read the next frame of audio data
    newSamples = fileReader();
    numNewSamples = length(newSamples);
    totalSamplesRead = totalSamplesRead + numNewSamples;

    % Shift buffer and add new samples
    buffer(1:end - stepSamples) = buffer(stepSamples + 1:end);
    buffer(end - stepSamples + 1:end) = newSamples;

    % Sync detection within the current frame
    if ~syncDetected
        for i = 1:numNewSamples
            if buffer(i) > startThreshold
                syncDetected = true;
                syncSamplePosition = totalSamplesRead - numNewSamples + i; % Sync position in the current frame
                fprintf('Sync signal detected at sample %d. Starting inference.\n', syncSamplePosition);
                break;
            end
        end
    end

    % Start inference only if sync signal is detected
    if syncDetected
        % Normalize buffer to range [-1, 1]
        maxVal = max(abs(buffer));
        if maxVal > 0
            segment = buffer / maxVal;
        else
            segment = buffer;
        end

        % Feature extraction
        [S, F, ~] = spectrogram(segment, hamming(windowSamples), overlap, nfft, Fs);
        meanSpectrum = mean(abs(S), 2);
        spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
        spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
        rmsValue = sqrt(mean(segment.^2));
        zeroCrossings = sum(abs(diff(sign(segment)))) / (2 * length(segment));
        coeffs = mfcc(segment, Fs, 'NumCoeffs', 13);
        meanMFCC = mean(coeffs, 1);

        % Pack features into feature vector
        featureVector = [spectralCentroid; spectralSpread; rmsValue; zeroCrossings; meanMFCC'];
        numFeatures = 100; % Ensure this matches your model's expected input size
        if length(featureVector) > numFeatures
            featureVector = featureVector(1:numFeatures);
        else
            featureVector = [featureVector; zeros(numFeatures - length(featureVector), 1)];
        end

        % Predict event label
        try
            [predictedLabel, score] = predict(model, featureVector');
        catch exception
            fprintf('Prediction error at segment starting %.2f seconds: %s\n', totalSamplesRead / Fs, exception.message);
            continue; % Skip this segment if there's an error
        end

        % Handle label output
        if iscell(predictedLabel) && ~isempty(predictedLabel)
            label = predictedLabel{1};
        elseif ischar(predictedLabel) || isstring(predictedLabel)
            label = char(predictedLabel);
        else
            fprintf('Unexpected label format at sample %d.\n', totalSamplesRead);
            continue;
        end

        % Check if the detected label meets the confidence threshold
        maxScore = max(score);
        if any(strcmp(label, {'N', 'F'})) && maxScore > eventThreshold
            refSound = strcmp(label, 'N') * refNormal + strcmp(label, 'F') * refFaulty;
            % Cross-correlation for accurate positioning
            [c, lag] = xcorr(buffer(1:round(0.1 * Fs)), refSound);
            [~, lagIdx] = max(c);
            accurateSamplePosition = totalSamplesRead - windowSamples + lag(lagIdx);
            accurateTime = accurateSamplePosition / Fs;
            
            % Suppress duplicate detections within the suppress period
            if (totalSamplesRead / Fs - lastEventTime) >= suppressTime
                savedSegment = buffer; 

                % Save the segment based on the label
                if strcmp(label, 'N')
                    filename = sprintf('Day2_dataset/Normal/%d.wav', eventCounter);
                    if ~exist('Normal_test', 'dir')
                        mkdir('Normal_test');
                    end
                    audiowrite(filename, savedSegment, Fs);

                elseif strcmp(label, 'F')
                    filename = sprintf('Day2_dataset/Faulty/%d.wav', eventCounter);
                    if ~exist('Faulty_test', 'dir')
                        mkdir('Faulty_test');
                    end
                    audiowrite(filename, savedSegment, Fs);
                end

                eventTimes(eventCounter) = totalSamplesRead / Fs;
                samplePositions(eventCounter) = accurateSamplePosition;
                eventLabels{eventCounter} = label;
                fprintf('Event detected at %.3f seconds (Sample Position: %d), Label: %s\n', ...
                        accurateTime, accurateSamplePosition, label);

                lastEventTime = accurateTime;
                eventCounter = eventCounter + 1;
            end
        else
            if ~exist('NonStamp_test', 'dir')
               mkdir('NonStamp_test');
            end
            savedSegment = buffer; 
            filename = sprintf('Day2_dataset/Non-Stamping/%d.wav', nonStampingCounter);
            nonStampingTime = totalSamplesRead / Fs;
            % fprintf('Non-stamping event detected at %.3f seconds\n', nonStampingTime);
            audiowrite(filename, savedSegment, Fs);
            nonStampingCounter = nonStampingCounter + 1;
        end
    end
end

release(fileReader);

eventTable = table(eventTimes', samplePositions', eventLabels', ...
                   'VariableNames', {'Time (s)', 'Sample Position', 'Event'});
disp(eventTable);
