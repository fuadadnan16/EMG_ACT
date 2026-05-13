%% 1. Parameters & Load
base_unix_time = 1774821794.060793;
audio_file = 'audio_1.wav';
[audioIn, fs] = audioread(audio_file);

%% 2. Custom Utterance Detection (Energy-Based)
% Convert to mono if stereo
if size(audioIn,2) > 1, audioIn = mean(audioIn,2); end

% Calculate RMS energy in 50ms frames
frameLen = round(0.050 * fs);
energy = sqrt(movmean(audioIn.^2, frameLen));

% --- TUNE THIS THRESHOLD ---
% Start with 0.02. If it misses quiet speech, lower it to 0.01 or 0.005.
threshold = 0.015 * max(energy); 

% Identify segments
isSpeech = energy > threshold;
changePoints = diff([0; isSpeech; 0]);
startIdx = find(changePoints == 1);
endIdx = find(changePoints == -1) - 1;

% Filter out segments shorter than 0.3s (noise/clicks)
valid = (endIdx - startIdx) > (0.3 * fs);
finalIdx = [startIdx(valid), endIdx(valid)];

%% 3. Visualization (CRITICAL for Accuracy)
% Use this to confirm the red lines cover every spoken word
figure;
t = (0:length(audioIn)-1)/fs;
plot(t, audioIn, 'Color', [0.7 0.7 0.7]); hold on;
plot(t, isSpeech * max(audioIn), 'r', 'LineWidth', 1.5);
title('Red Line = Detected Utterance. Adjust "threshold" if wrong.');
xlabel('Time (seconds)');

%% 4. Transcription Logic
% Since R2018a lacks 'speech2text', you must use an external API 
% or the Google Speech API via webread if you have a key.
% Here, we calculate the TIMESTAMPS for your CSV.

results = cell(size(finalIdx, 1), 5); % Table-like cell array

for i = 1:size(finalIdx, 1)
    % Relative Time
    rel_start = (finalIdx(i,1)-1) / fs;
    rel_end = (finalIdx(i,2)-1) / fs;
    
    % Unix Time
    unix_start = base_unix_time + rel_start;
    unix_end = base_unix_time + rel_end;
    
    % Placeholder for Text (Since 2018a doesn't have built-in Whisper)
    % Suggestion: Export these segments and use a modern tool for the text.
    txt = "Utterance_" + i; 
    
    results(i, :) = {txt, rel_start, rel_end, unix_start, unix_end};
end

%% 5. Save to CSV
header = {'Text', 'Audio_Start_Sec', 'Audio_End_Sec', 'Unix_Start', 'Unix_End'};
outputTable = cell2table(results, 'VariableNames', header);
writetable(outputTable, 'utterance_timestamps.csv');

fprintf('Detected %d utterances. Data saved to CSV.\n', size(finalIdx,1));