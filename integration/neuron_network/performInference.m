function predictedLabel = performInference(segment, layer_params, windowLength, fftLength, labelNames)
    %#codegen

    % Normalize the input segment
    if max(abs(segment)) > 0
        segment = segment / max(abs(segment)); % Normalize to [-1, 1]
    end

    % Perform FFT on the normalized segment
    fftResult = fft(segment, windowLength); % Perform FFT
    features = abs(fftResult(1:fftLength)).'; % Take magnitude and transpose for 1-by-fftLength

    % Perform inference using the extracted model parameters
    activations = features;  % Initialize activations with FFT features

    % Forward pass through each layer using weights and biases
    for i = 1:numel(layer_params)
        if isfield(layer_params(i), 'Weights') && isfield(layer_params(i), 'Bias')
            % Fully connected layer: activations = (activations * Weights) + Bias
            activations = activations * layer_params(i).Weights + layer_params(i).Bias;

            % Apply ReLU activation if the layer type is ReLU
            if isfield(layer_params(i), 'Type') && strcmp(layer_params(i).Type, 'nnet.cnn.layer.ReLULayer')
                activations = max(0, activations); % ReLU
            end
        end
    end

    % Assuming output is a classification result
    [~, predictedIndex] = max(activations); % Choose class with highest activation
    predictedLabel = labelNames(predictedIndex); % Map index to label
end
