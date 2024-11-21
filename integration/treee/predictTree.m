function [predictedLabel, confidence] = predictTree(segment, treeModel, fftLength, labelNames)
    % Predict the class label of an audio segment using a decision tree model
    % and return confidence in the prediction.
    % Inputs:
    %   segment - 1D vector of audio data for a 1-second window
    %   treeModel - Trained decision tree model
    %   fftLength - Length of the FFT feature vector (half-spectrum length)
    %   labelNames - Array of label names for output mapping
    % Outputs:
    %   predictedLabel - Predicted class label
    %   confidence - Confidence score for the prediction (posterior probability)
    
    % Compute FFT of the segment
    fftResult = fft(segment, length(segment));
    features = abs(fftResult(1:fftLength)).';

    % Ensure the feature vector has the correct length
    if length(features) < fftLength
        features = [features, zeros(1, fftLength - length(features))];
    elseif length(features) > fftLength
        features = features(1:fftLength);
    end

    % Predict class label and posterior probabilities for the current window
    [predictedLabelIndex, scores] = predict(treeModel, features);
    predictedLabel = labelNames(predictedLabelIndex);

    % Extract the confidence score (posterior probability) of the predicted class
    confidence = scores(predictedLabelIndex);
end
