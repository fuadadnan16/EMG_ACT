%% generate_ground_truth.m (Clean Transcription + Manual Label Template)
clear; clc;

%% ──────────── SETTINGS ────────────
audio_wav    = 'audio_1.wav';
audio_info   = 'audio_info_1.csv';
output_file  = 'ground_truth_final_utterence.csv';
plot_file    = 'utterance_verification.png';
whisper_model = 'large';

vad_threshold    = 0.001;
vad_merge_gap_s  = 0.5;
vad_min_dur_s    = 0.2;

label_names = {'palm','thumb','index_tip','index_seg', ...
               'middle_tip','middle_seg','ring_tip','ring_seg', ...
               'pinky_tip','pinky_seg'};

%% ──────────── STEP 1: LOAD AUDIO & DETECT ────────────
fprintf('[Step 1] Loading audio: %s\n', audio_wav);
[audio, fs] = audioread(audio_wav);
if size(audio, 2) > 1, audio = mean(audio, 2); end

info = readtable(audio_info);
audio_start_unix = info.audio_start_unix_time_s(1);

idx = detectSpeech(audio, fs, 'MergeDistance', fs*vad_merge_gap_s);
durations = (idx(:,2) - idx(:,1)) / fs;
idx = idx(durations >= vad_min_dur_s, :);
n_segments = size(idx, 1);
fprintf('         Found %d speech segments\n', n_segments);

%% ──────────── STEP 2: WHISPER TRANSCRIPTION (via Python) ────────────
fprintf('[Step 2] Transcribing each segment with Whisper %s...\n', whisper_model);

% Save each segment as a temp wav, transcribe with Python Whisper
tmp_dir = 'tmp_segments';
if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end

% Write all segments as separate wav files
for i = 1:n_segments
    seg_audio = audio(idx(i,1):idx(i,2));
    audiowrite(fullfile(tmp_dir, sprintf('seg_%04d.wav', i)), seg_audio, fs);
end

% Write Python script that transcribes all segments
tmp_py   = 'tmp_transcribe.py';
tmp_json = 'tmp_results.json';

fid = fopen(tmp_py, 'w');
fprintf(fid, 'import whisper, json, warnings, os, glob\n');
fprintf(fid, 'warnings.filterwarnings("ignore")\n');
fprintf(fid, 'os.environ["TOKENIZERS_PARALLELISM"] = "false"\n');
fprintf(fid, 'model = whisper.load_model("%s")\n', whisper_model);
fprintf(fid, 'prompt = (\n');
fprintf(fid, '    "Short commands about hand segments. "\n');
fprintf(fid, '    "Numbers: zero one two three four five six seven eight nine. "\n');
fprintf(fid, '    "Phrases: whole hand, all hand, all fingertips, no pinky, no thumb, "\n');
fprintf(fid, '    "no index, nothing, rest. Combinations: one and two, one two four six."\n');
fprintf(fid, ')\n');
fprintf(fid, 'results = {}\n');
fprintf(fid, 'files = sorted(glob.glob("%s/seg_*.wav"))\n', strrep(tmp_dir, '\', '/'));
fprintf(fid, 'for f in files:\n');
fprintf(fid, '    name = os.path.basename(f)\n');
fprintf(fid, '    r = model.transcribe(f, language="en", temperature=0.0, beam_size=5,\n');
fprintf(fid, '        best_of=5, initial_prompt=prompt, verbose=False)\n');
fprintf(fid, '    text = r["text"].strip() if r["text"] else ""\n');
fprintf(fid, '    results[name] = text\n');
fprintf(fid, '    print(f"  {name}: {text}")\n');
fprintf(fid, 'with open("%s", "w") as f:\n', tmp_json);
fprintf(fid, '    json.dump(results, f, indent=2)\n');
fprintf(fid, 'print("WHISPER_DONE")\n');
fclose(fid);

[status, res] = system(sprintf('python3 %s 2>&1', tmp_py));
delete(tmp_py);

if status ~= 0 || ~contains(res, 'WHISPER_DONE')
    % Clean up temp files
    rmdir(tmp_dir, 's');
    error('Whisper failed:\n%s', res);
end

% Read results
json_str = fileread(tmp_json);
delete(tmp_json);
whisper_results = jsondecode(json_str);

% Clean up temp segment files
rmdir(tmp_dir, 's');

%% ──────────── BUILD RESULTS TABLE ────────────
fprintf('[Step 3] Building results table...\n');

results = cell(n_segments, 8 + numel(label_names));

for i = 1:n_segments
    start_s = (idx(i,1) - 1) / fs;
    end_s   = (idx(i,2) - 1) / fs;
    dur_s   = end_s - start_s;
    unix_s  = audio_start_unix + start_s;
    unix_e  = audio_start_unix + end_s;
    
    % Get Whisper text for this segment
    seg_name = sprintf('seg_%04d.wav', i);
    if isfield(whisper_results, strrep(seg_name, '.', '_'))
        % jsondecode may replace . with _ in field names
        txt = string(whisper_results.(strrep(seg_name, '.', '_')));
    elseif isfield(whisper_results, seg_name)
        txt = string(whisper_results.(seg_name));
    else
        txt = "";
    end
    
    % Determine status
    if strlength(txt) > 0
        status = 'MATCHED';
    else
        status = 'UNMATCHED';
    end
    
    results(i, 1:8) = {i, start_s, end_s, dur_s, unix_s, unix_e, status, txt};
    results(i, 9:end) = num2cell(zeros(1, numel(label_names)));
end

%% ──────────── STEP 3: EXPORT AND PLOT ────────────
headers = [{'segment','start_s','end_s','duration_s','start_unix','end_unix','status','text'}, ...
           strcat('label_', label_names)];
outputTable = cell2table(results, 'VariableNames', headers);
writetable(outputTable, output_file);

% Save verification plot
f = figure('visible', 'off', 'Position', [50 50 2000 500]);
t = (0:length(audio)-1)/fs;
plot(t, audio, 'Color', [0.8 0.8 0.8]); hold on;

for i = 1:n_segments
    s = (idx(i,1)-1)/fs;
    e = (idx(i,2)-1)/fs;
    x = [s e e s];
    y = [-1 -1 1 1] * max(abs(audio)) * 0.9;
    
    if strcmp(results{i,7}, 'MATCHED')
        patch(x, y, 'green', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    else
        patch(x, y, 'red', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    
    mid = (s+e)/2;
    lbl = sprintf('#%d', i);
    if strlength(string(results{i,8})) > 0
        lbl = sprintf('#%d: %s', i, string(results{i,8}));
    end
    text(mid, max(abs(audio))*0.95, lbl, 'FontSize', 5, ...
         'HorizontalAlignment', 'center', 'Rotation', 60);
end

xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Green=MATCHED (%d) | Red=UNMATCHED (%d)', ...
      sum(strcmp(results(:,7),'MATCHED')), sum(strcmp(results(:,7),'UNMATCHED'))));
hold off;
saveas(f, plot_file);
close(f);

fprintf('[Done] %s (%d segments) and %s saved.\n', output_file, n_segments, plot_file);
fprintf('       MATCHED: %d | UNMATCHED: %d\n', ...
        sum(strcmp(results(:,7),'MATCHED')), sum(strcmp(results(:,7),'UNMATCHED')));
fprintf('       Fill in text+labels for UNMATCHED rows in Excel.\n');
