function downloadAndPlayAudioFromApi(apiUrl, name)
    % downloadAndPlayAudioFromApi Sends a POST request to get a Google Drive
    % URL, downloads the audio file, and plays it at the original sampling rate.
    %
    % Inputs:
    %   - apiUrl : The API URL that provides the Google Drive URL.
    %   - name   : The name to be included in the JSON body of the POST request.
    %
    % Example:
    %   downloadAndPlayAudioFromApi('https://example.com/api', 'audioFileName')

    % Step 1: Send POST request with the name in the JSON body
    data = struct('name', name); % Prepare the data structure with the name field
    options = weboptions('MediaType', 'application/json');
    response = webwrite(apiUrl, data, options);

    % Extract the Google Drive URL from the response
    if isfield(response, 'driveUrl')
        driveUrl = response.driveUrl;
    else
        error('The response does not contain a driveUrl field.');
    end

    % Step 2: Convert Google Drive link to a direct download link
    fileID = extractBetween(driveUrl, "/d/", "/");
    if isempty(fileID)
        error('Invalid Google Drive URL format.');
    end
    downloadUrl = ['https://drive.google.com/uc?export=download&id=', fileID{1}];

    % Step 3: Download the file
    outputFilename = 'downloaded_audio.wav';
    websave(outputFilename, downloadUrl);
    disp(['File downloaded as ', outputFilename]);

    % Step 4: Read and play the audio file
    [audioData, sampleRate] = audioread(outputFilename);
    disp(['Sample rate: ', num2str(sampleRate), ' Hz']);
    sound(audioData, sampleRate); % Plays the audio at the original sampling rate
end
