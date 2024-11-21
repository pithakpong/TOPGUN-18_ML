% Define example inputs for codegen
segment = zeros(windowLength, 1);  % Example segment
fftLength = windowLength / 2 + 1;  % Number of FFT bins
labelNames = ["N", "F", "X"];  % Example labels

% Run codegen command
codegen -config cfg performInference -args {segment, coder.Constant(layer_params), windowLength, fftLength, coder.Constant(labelNames)}
