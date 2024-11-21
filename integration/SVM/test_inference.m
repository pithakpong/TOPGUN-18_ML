% Parameters for testing
inputFile = 'Day2/Day2.wav'; % Long audio file for testing
windowLength = 1.0; % Window length in seconds
overlapRatio = 0.9; % 90% overlap for higher detection frequency
Fs = 48000; % Sampling rate of the audio file
fftLength = Fs / 2 + 1; % Half-spectrum FFT length for 1-second window
stepSamples = round(windowLength * Fs * (1 - overlapRatio)); % Step size in samples

% Define label names (adjust to match the classes in your SVM model)
labelNames = ["N", "F", "X"];

% Load and normalize the audio file
[x, Fs] = audioread(inputFile);
x = double(x(:)); % Ensure x is a column vector and of type double
x = x / max(abs(x)); % Normalize entire audio file to range [-1, 1]

% Initialize buffer for the sliding window
windowSamples = round(windowLength * Fs);
currentSample = 1;

% Process the audio file with window slicing and overlap
while currentSample + windowSamples <= length(x)
    % Extract the current window of audio
    segment = x(currentSample : currentSample + windowSamples - 1);

    % Normalize the segment
    maxVal = max(abs(segment));
    if maxVal > 0
        segment = segment / maxVal;
    end

    % Call predictSegmentSVM function for prediction
    [predictedLabel, confidence] = predictSegmentSVM(segment, svmModel, fftLength, labelNames);

    % Calculate the current window start time in seconds
    currentTime = currentSample / Fs;

    % Display the result
    fprintf('Time: %.2f sec - Predicted Label: %s - Confidence: %.2f\n', currentTime, predictedLabel, confidence);

    % Move to the next segment based on the overlap
    currentSample = currentSample + stepSamples;
end

fprintf('End of file reached.\n');
