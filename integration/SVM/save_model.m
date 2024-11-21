% Extract and save parameters
numBinaryModels = numel(svmModel.BinaryLearners);
weights = cell(1, numBinaryModels);
biases = cell(1, numBinaryModels);

for i = 1:numBinaryModels
    % Ensure each learner is linear
    if strcmp(svmModel.BinaryLearners{i}.KernelFunction, 'linear')
        weights{i} = svmModel.BinaryLearners{i}.Beta;
        biases{i} = svmModel.BinaryLearners{i}.Bias;
    else
        error('Non-linear SVMs are not supported in this example.');
    end
end

% Save weights and biases for code generation
save('svm_parameters.mat', 'weights', 'biases');
