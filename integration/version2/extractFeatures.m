function featureVector = extractFeatures(x, Fs)
    % Normalize audio signal to the range [-1, 1]
    x = x / max(abs(x));
    
    % Parameters for spectrogram and MFCC
    windowLength = 256;
    overlap = 128;
    nfft = 512;
    numFeatures = 100; % Desired number of features to match model input

    % Compute the spectrogram
    [S, F, ~] = spectrogram(x, hamming(windowLength), overlap, nfft, Fs);
    
    % Feature calculations
    meanSpectrum = mean(abs(S), 2);
    spectralCentroid = sum(F .* meanSpectrum) / sum(meanSpectrum);
    spectralSpread = sqrt(sum((F - spectralCentroid).^2 .* meanSpectrum) / sum(meanSpectrum));
    rmsValue = sqrt(mean(x.^2));
    zeroCrossings = sum(abs(diff(sign(x)))) / (2 * length(x));
    
    % Compute MFCCs and take the mean of each coefficient
    coeffs = mfcc(x, Fs, 'NumCoeffs', 13);
    meanMFCC = mean(coeffs, 1);
    
    % Pack features into feature vector
    rawFeatureVector = [spectralCentroid; spectralSpread; rmsValue; zeroCrossings; meanMFCC'];
    
    % Adjust feature vector to have exactly 100 elements
    if length(rawFeatureVector) > numFeatures
        featureVector = rawFeatureVector(1:numFeatures); % Truncate to 100 elements
    else
        featureVector = [rawFeatureVector; zeros(numFeatures - length(rawFeatureVector), 1)]; % Pad with zeros
    end
end
