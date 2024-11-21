% Parameters for real-time prediction
Fs = 48000; % Sampling rate
inputFile = 'Day2/Day2.wav'; % Long audio file for testing
windowLength = round(1.0 * Fs); % 1-second window length in samples
overlapRatio = 0.9; % 90% overlap
fftLength = windowLength / 2 + 1; % Half spectrum length for FFT output
stepSamples = round(windowLength * (1 - overlapRatio)); % Step size in samples

% Define label names
labelNames = ["N", "F", "X"]; % Align these labels with your model classes

% Initialize audio file reader
fileReader = dsp.AudioFileReader(inputFile, ...
    'SamplesPerFrame', stepSamples, ...
    'PlayCount', 1);

% Initialize buffer for the sliding window
buffer = zeros(windowLength, 1); % Store each 1-second window
sampleCounter = 0;

% Real-time processing loop
while ~isDone(fileReader)
    % Read the next segment of audio data
    newSamples = fileReader();
    numNewSamples = length(newSamples);
    sampleCounter = sampleCounter + numNewSamples;

    % Shift buffer and add new samples
    buffer(1:end - numNewSamples) = buffer(numNewSamples + 1:end);
    buffer(end - numNewSamples + 1:end) = newSamples;

    % Perform inference
    predictedLabel = performInference(buffer, net, windowLength, fftLength, labelNames);

    % Calculate the current time in seconds
    currentTime = sampleCounter / Fs;
    fprintf('Time: %.2f sec - Predicted Label: %s\n', currentTime, char(predictedLabel));
end

% Release the audio file reader
release(fileReader);
