cfg = coder.config('lib');          % Generate a static library (.a file)
cfg.Hardware = raspiConfig;         % Assign the Raspberry Pi hardware configuration

% Example codegen command
codegen -config cfg predictSegmentSVM -args {segment, coder.Constant(weights), coder.Constant(biases), fftLength, coder.Constant(labelNames)}
