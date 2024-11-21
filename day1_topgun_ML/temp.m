[y, fs] = audioread('Day1/Normal/Norm_2_52_pm_3.wav');

% Play the audio
player = audioplayer(y, fs);
play(player);