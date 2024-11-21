% List training files
normalFiles = dir("Day2_dataset/Normal/*.wav");
faultyFiles = dir("Day2_dataset/Faulty/*.wav");
nonStampingFiles = dir("Day2_dataset/Non-Stamping/*.wav");
numNormalSamples = length(normalFiles);
numFaultySamples = length(faultyFiles);
numNonStampingSamples = length(nonStampingFiles);

% Prepare input and label matrices
xFiles = [normalFiles; faultyFiles; nonStampingFiles];
y = [repmat({'N'}, numNormalSamples, 1);
     repmat({'F'}, numFaultySamples, 1);
     repmat({'X'}, numNonStampingSamples, 1)];

% Shuffle X and y
numSamples = length(y);
idx = randperm(numSamples);
xFiles = xFiles(idx, :);
y = y(idx);

% Define feature extraction parameters
numFeatures = 100;
windowLength = 256;
overlap = 128;
nfft = 512;

% Preallocate feature matrix
X = zeros(numSamples, numFeatures);

% Feature extraction loop with normalization
for i = 1:numSamples
    % Load audio file based on class
    switch y{i}
        case 'N'
            [x, Fs] = audioread("Day2_dataset/Normal/" + xFiles(i).name);
        case 'F'
            [x, Fs] = audioread("Day2_dataset/Faulty/" + xFiles(i).name);
        case 'X'
            [x, Fs] = audioread("Day2_dataset/Non-Stamping/" + xFiles(i).name);
    end
    
    % Normalize audio signal to the range [-1, 1]
    x = x / max(abs(x));
    
    % Feature extraction
    [S, F, T] = spectrogram(x, hamming(windowLength), overlap, nfft, Fs);
    meanSpectrum = mean(abs(S), 2);
    spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
    spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
    rmsValue = sqrt(mean(x.^2));
    zeroCrossings = sum(abs(diff(sign(x)))) / (2 * length(x));
    coeffs = mfcc(x, Fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);
    
    % Pack features into feature vector
    featureVector = [spectralCentroid; spectralSpread; rmsValue; zeroCrossings; meanMFCC'];
    if length(featureVector) > numFeatures
        featureVector = featureVector(1:numFeatures);
    else
        featureVector = [featureVector; zeros(numFeatures - length(featureVector), 1)];
    end
    
    X(i, :) = featureVector';
end

% Split data into training, validation, and test sets (stratified by class)
cv = cvpartition(y, 'HoldOut', 0.4); % 60% training, 40% test-validation
XTrainVal = X(cv.training, :);
yTrainVal = y(cv.training);
XTest = X(cv.test, :);
yTest = y(cv.test);

% Further split training-validation set into 75% training and 25% validation
cvVal = cvpartition(yTrainVal, 'HoldOut', 0.25);
XTrain = XTrainVal(cvVal.training, :);
yTrain = yTrainVal(cvVal.training);
XVal = XTrainVal(cvVal.test, :);
yVal = yTrainVal(cvVal.test);

% Train k-NN model on training set
rng(42); % Set random seed for reproducibility
numNeighbors = 5; % Define the number of neighbors
model = fitcknn(XTrain, yTrain, 'NumNeighbors', numNeighbors, 'Standardize', true);

% Validate model on validation set
yValPred = predict(model, XVal);
valConfMat = confusionmat(yVal, yValPred);
valAccuracy = sum(strcmp(yValPred, yVal)) / length(yVal);

% Evaluate model on test set
yTestPred = predict(model, XTest);
testConfMat = confusionmat(yTest, yTestPred);
testAccuracy = sum(strcmp(yTestPred, yTest)) / length(yTest);

% Display results
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 100);
disp('Validation Confusion Matrix:');
disp(valConfMat);

fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);
disp('Test Confusion Matrix:');
disp(testConfMat);

% Process unknown files
unknownFiles = dir("Day1/Unknown/*.wav");
unknownFilenames = struct2table(unknownFiles);
unknownFilenames = sortrows(unknownFilenames, 'name');
unknownFiles = table2struct(unknownFilenames);

% Preallocate prediction results
predictions = cell(length(unknownFiles), 1);

% Process each unknown file
for i = 1:length(unknownFiles)
    % Load audio
    [x, Fs] = audioread("Day1/Unknown/" + unknownFiles(i).name);
    
    % Normalize audio signal to the range [-1, 1]
    x = x / max(abs(x));
    
    % Extract features (same as training pipeline)
    [S, F, T] = spectrogram(x, hamming(windowLength), overlap, nfft, Fs);
    meanSpectrum = mean(abs(S), 2);
    spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
    spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
    rmsValue = sqrt(mean(x.^2));
    zeroCrossings = sum(abs(diff(sign(x)))) / (2 * length(x));
    coeffs = mfcc(x, Fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);
    
    % Pack features
    featureVector = [spectralCentroid; spectralSpread; rmsValue; zeroCrossings; meanMFCC'];
    if length(featureVector) > numFeatures
        featureVector = featureVector(1:numFeatures);
    else
        featureVector = [featureVector; zeros(numFeatures - length(featureVector), 1)];
    end
    
    % Make prediction
    prediction = predict(model, featureVector');
    predictions{i} = prediction{1};
end

% Concatenate predictions with space separation and print result
predictedLabelsStr = strjoin(predictions, ' ');
fprintf('%s\n', predictedLabelsStr);
save('knn_model.mat','model')
