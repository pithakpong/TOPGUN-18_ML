% Define parameters
Fs = 48000; % Sampling rate
windowLength = Fs; % 1 second of audio data (48000 samples)
fftLength = windowLength / 2 + 1; % Half spectrum (FFT is symmetric)
numClasses = 3; % Number of classes (e.g., 'Normal', 'Faulty', 'Non-Stamping')

% Load data files for each class
normalFiles = dir("Day2_dataset/Normal/*.wav");
faultyFiles = dir("Day2_dataset/Faulty/*.wav");
nonStampingFiles = dir("Day2_dataset/Non-Stamping/*.wav");

% Assign labels
numNormal = length(normalFiles);
numFaulty = length(faultyFiles);
numNonStamping = length(nonStampingFiles);
y = [ones(numNormal, 1); 2 * ones(numFaulty, 1); 3 * ones(numNonStamping, 1)]; % Label each class as 1, 2, 3

% Initialize feature and label arrays
numSamples = numNormal + numFaulty + numNonStamping;
X = zeros(numSamples, fftLength);
yCategorical = categorical(y);

% FFT feature extraction
for i = 1:numSamples
    if i <= numNormal
        [x, ~] = audioread("Day2_dataset/Normal/" + normalFiles(i).name);
    elseif i <= numNormal + numFaulty
        [x, ~] = audioread("Day2_dataset/Faulty/" + faultyFiles(i - numNormal).name);
    else
        [x, ~] = audioread("Day2_dataset/Non-Stamping/" + nonStampingFiles(i - numNormal - numFaulty).name);
    end

    % Ensure 1-second length by trimming or padding
    if length(x) > windowLength
        x = x(1:windowLength);
    elseif length(x) < windowLength
        x = [x; zeros(windowLength - length(x), 1)];
    end

    % Normalize and compute FFT
    x = x / max(abs(x)); % Normalize
    fftResult = fft(x, windowLength); % Perform FFT
    X(i, :) = abs(fftResult(1:fftLength)).'; % Take the magnitude and transpose to get 1-by-24001
end

% Split data into training and validation sets
cv = cvpartition(y, 'HoldOut', 0.2); % 80% training, 20% validation
XTrain = X(training(cv), :);
yTrain = y(training(cv)); % SVM uses numerical labels, not categorical
XVal = X(test(cv), :);
yVal = y(test(cv));

% Train the multiclass SVM model using ECOC
% Train an ECOC model for multiclass SVM
t = templateSVM('KernelFunction', 'linear', 'Standardize', true);
svmModel = fitcecoc(XTrain, yTrain, 'Learners', t);
% Evaluate the model on the validation set
yValPred = predict(svmModel, XVal);

% Display confusion matrix
valConfMat = confusionmat(yVal, yValPred);
disp('Validation Confusion Matrix:');
disp(valConfMat);

% Plot the confusion matrix as a chart
figure;
confusionchart(yVal, yValPred);
title('Confusion Matrix for Validation Set');
