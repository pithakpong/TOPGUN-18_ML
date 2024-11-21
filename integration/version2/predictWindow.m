function label = predictWindow(features)
    % Load the pre-trained model using loadLearnerForCoder
    model = loadLearnerForCoder('trainedModel');

    % Make a prediction
    label = predict(model, features');
end
