% Load saved model parameters (weights, biases) from the .mat file
load('model_parameters.mat');  % This file should contain 'layer_params'

% Define example inputs
windowLength = 48000;  % Adjust based on your audio settings
fftLength = windowLength / 2 + 1;  % Number of FFT bins
segment = zeros(windowLength, 1);  % Example 1-second audio segment
labelNames = ["N", "F", "X"];  % Example label array

cfg = coder.config('lib');  % Generate a static library (.a file)
cfg.Hardware = coder.hardware('Raspberry Pi'); % Set target hardware as Raspberry Pi
codegen -config cfg performInference -args {segment, coder.Constant(layer_params), windowLength, fftLength, coder.Constant(labelNames)}
