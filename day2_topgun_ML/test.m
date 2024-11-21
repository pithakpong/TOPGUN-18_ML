% Parameters
frameLength = 256; % Number of samples per frame

% Initialize Audio File Reader
fileReader = dsp.AudioFileReader('Day2/Normal1.wav', 'SamplesPerFrame', frameLength);

% Get the sample rate from the audio file
Fs = fileReader.SampleRate;

% Initialize Audio Device Writer to play audio
deviceWriter = audioDeviceWriter('SampleRate', Fs);

% Playback loop
while ~isDone(fileReader)
    % Read a frame of audio data
    audioFrame = fileReader();
    
    % Play the audio frame
    deviceWriter(audioFrame);
end

% Release system resources
release(fileReader);
release(deviceWriter);
