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

% Feature extraction loop
for i = 1:numSamples
    % Load audio file based on class
    switch y{i}
        case 'N'
            [x, Fs] = audioread("Day2_dataset/Normal/" + xFiles(i).name);
            fprintf('%s\n', xFiles(i).name);
        case 'F'
            [x, Fs] = audioread("Day2_dataset/Faulty/" + xFiles(i).name);
            fprintf('%s\n', xFiles(i).name);
        case 'X'
            [x, Fs] = audioread("Day2_dataset/Non-Stamping/" + xFiles(i).name);
            fprintf('%s\n', xFiles(i).name);
    end
    % Feature extraction
    [S, F, T] = spectrogram(x, hamming(windowLength), overlap, nfft, Fs);
    meanSpectrum = mean(abs(S), 2);
    spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
    spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
    rms = sqrt(mean(x.^2));
    zeroCrossings = sum(abs(diff(sign(x)))) / (2 * length(x));
    coeffs = mfcc(x, Fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);
    
    % Pack features into feature vector
    featureVector = [spectralCentroid; spectralSpread; rms; zeroCrossings; meanMFCC'];
    if length(featureVector) > numFeatures
        featureVector = featureVector(1:numFeatures);
    else
        featureVector = [featureVector; zeros(numFeatures - length(featureVector), 1)];
    end
    
    X(i, :) = featureVector';
end

% Split data into training and validation sets
cv = cvpartition(numSamples, 'HoldOut', 0.2);
XTrain = X(cv.training, :);
yTrain = y(cv.training);
XVal = X(cv.test, :);
yVal = y(cv.test);

% Train Random Forest model
rng(42); % Set random seed for reproducibility
numTrees = 100;
model = TreeBagger(numTrees, XTrain, yTrain, 'Method', 'classification');

% Validate model (Optional)
yPred = predict(model, XVal);
confMat = confusionmat(yVal, yPred);
accuracy = sum(strcmp(yPred, yVal)) / length(yVal);

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
    
    % Extract features (same as training pipeline)
    [S, F, T] = spectrogram(x, hamming(windowLength), overlap, nfft, Fs);
    meanSpectrum = mean(abs(S), 2);
    spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
    spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
    rms = sqrt(mean(x.^2));
    zeroCrossings = sum(abs(diff(sign(x)))) / (2 * length(x));
    coeffs = mfcc(x, Fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);
    
    % Pack features
    featureVector = [spectralCentroid; spectralSpread; rms; zeroCrossings; meanMFCC'];
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
