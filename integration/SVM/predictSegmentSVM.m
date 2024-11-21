function [predictedLabel, confidence] = predictSegmentSVM(segment, weights, biases, fftLength, labelNames)
    %#codegen

    % Set the length of the FFT based on the segment length
    windowSamples = length(segment);

    % Perform FFT on the segment and extract magnitude as features
    fftResult = fft(segment, windowSamples); % Perform FFT
    features = abs(fftResult(1:fftLength)).'; % Take magnitude and transpose to 1-by-fftLength

    % Initialize scores for each class (for multiclass SVM)
    numClasses = numel(labelNames);
    scores = zeros(1, numClasses);

    % Perform binary classification for each model in the ECOC
    for i = 1:numel(weights)
        % Calculate the decision score for each binary SVM
        decisionScore = features * weights{i} + biases{i};
        
        % Vote for classes based on decision score
        if decisionScore >= 0
            scores(1) = scores(1) + 1; % Positive class
        else
            scores(2) = scores(2) + 1; % Negative class
        end
    end

    % Determine the predicted class based on majority voting
    [confidence, predictedLabelIdx] = max(scores);
    predictedLabel = labelNames{predictedLabelIdx}; % Map index to label
end
