% Parameters
normalFiles = dir("Split_datasets/N/*.wav");
faultyFiles = dir("Split_datasets/F/*.wav");
nonStampingFiles = dir("Split_datasets/NS/*.wav");

numNormalSamples = length(normalFiles);
numFaultySamples = length(faultyFiles);
numNonStampingSamples = length(nonStampingFiles);

% Labels for each class
y = [repmat("N", numNormalSamples, 1);
     repmat("F", numFaultySamples, 1);
     repmat("X", numNonStampingSamples, 1)];

% Store features and labels
X = {};
Fs = 48000;

% Define feature extraction parameters
numMFCC = 13;
windowLength = 256;
overlap = 128;
nfft = 512;

% Feature extraction loop
allFiles = [normalFiles; faultyFiles; nonStampingFiles];
for i = 1:length(allFiles)
    [x, ~] = audioread(fullfile(allFiles(i).folder, allFiles(i).name));
    x = x / max(abs(x));
    % Replace 'WindowLength' and 'OverlapLength' with 'Window' and 'OverlapLength'
    coeffs = mfcc(x, Fs, 'Window', hamming(windowLength), 'OverlapLength', overlap, 'NumCoeffs', numMFCC);

    X{i} = coeffs';
end

% Convert labels to categorical
y = categorical(y);

featureDimension = 14; % Adjust this to the feature dimension of your training data
numHiddenUnits = 100;  % Example number of hidden units
numClasses = 3;        % Number of classes for classification

layers = [
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Split data into training and validation sets
cv = cvpartition(y, 'HoldOut', 0.2);
XTrain = X(training(cv));
yTrain = y(training(cv));
XVal = X(test(cv));
yVal = y(test(cv));

% Train the network
net = trainNetwork(XTrain, yTrain, layers, options);

% Predict on validation data
yValPred = classify(net, XVal);
valAccuracy = sum(yValPred == yVal) / numel(yVal) * 100;

% Display results
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy);
confMat = confusionmat(yVal, yValPred);
disp('Validation Confusion Matrix:');
disp(confMat);

% Save the model
save('rnn_model.mat', 'net');
