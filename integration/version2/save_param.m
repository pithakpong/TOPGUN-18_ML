% Extract and save model parameters for use in standalone code

% Define the file to save the model parameters
outputFile = 'modelParams.mat';

% Extract weights and biases from each layer
fc1Weights = net.Layers(2).Weights; % Fully connected layer 1 weights
fc1Bias = net.Layers(2).Bias;       % Fully connected layer 1 bias

fc2Weights = net.Layers(4).Weights; % Fully connected layer 2 weights
fc2Bias = net.Layers(4).Bias;       % Fully connected layer 2 bias

fc3Weights = net.Layers(6).Weights; % Fully connected layer 3 weights
fc3Bias = net.Layers(6).Bias;       % Fully connected layer 3 bias

% Save all parameters to a .mat file
save(outputFile, 'fc1Weights', 'fc1Bias', 'fc2Weights', 'fc2Bias', 'fc3Weights', 'fc3Bias');
