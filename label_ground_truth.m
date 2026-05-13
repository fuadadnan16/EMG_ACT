%% label_ground_truth.m
%  Reads ground_truth_final_utterence.csv and fills in the 10 label columns.
%  Keep this separate from the transcription script.
%
%  Hand mapping:
%    0=palm  1=thumb  2=index_tip  3=index_seg  4=middle_tip
%    5=middle_seg  6=ring_tip  7=ring_seg  8=pinky_tip  9=pinky_seg
%
%  Usage:
%    matlab -nodisplay -nosplash -r "label_ground_truth; exit"

clear; clc;

input_file  = 'ground_truth_final_utterence.csv';
output_file = 'ground_truth_labeled.csv';

%% ──────────── READ CSV ────────────
fprintf('Reading: %s\n', input_file);
T = readtable(input_file, 'TextType', 'string');
n = height(T);
fprintf('Found %d segments\n\n', n);

%% ──────────── LABEL EACH ROW ────────────
label_cols = {'label_palm','label_thumb','label_index_tip','label_index_seg', ...
              'label_middle_tip','label_middle_seg','label_ring_tip','label_ring_seg', ...
              'label_pinky_tip','label_pinky_seg'};

for i = 1:n
    txt = lower(strtrim(T.text(i)));
    txt = regexprep(txt, '[^\w\s]', '');
    txt = strtrim(regexprep(txt, '\s+', ' '));

    labels = parse_command(txt);

    for j = 1:10
        T.(label_cols{j})(i) = labels(j);
    end

    % Print for verification
    lbl_str = sprintf('%d', labels);
    fprintf('%3d  %-45s  %s\n', i, T.text(i), lbl_str);
end

%% ──────────── SAVE ────────────
writetable(T, output_file);
fprintf('\nSaved: %s\n', output_file);
fprintf('Done! Open in Excel to verify.\n');


%% ════════════════════════════════════════════
function labels = parse_command(text)
    labels = zeros(1, 10);
    % MATLAB index:  1=palm 2=thumb 3=idx_tip 4=idx_seg 5=mid_tip
    %                6=mid_seg 7=ring_tip 8=ring_seg 9=pinky_tip 10=pinky_seg

    if text == "" || text == " "
        return;
    end

    % Fix mishearings
    text = strrep(text, 'almond fingertips', 'all fingertips');
    text = strrep(text, 'all of hands', 'whole hand');
    text = strrep(text, 'all of hand', 'whole hand');
    text = strrep(text, 'whole of hand', 'whole hand');
    text = strrep(text, 'and pause', 'and palm');
    text = strrep(text, 'no fingertips except', 'all fingertips except');

    % ── Nothing / rest ──
    if contains(text, 'nothing') || contains(text, 'rest') || contains(text, 'none')
        return;
    end

    % ── "No X" = whole hand EXCEPT X ──
    no_match = regexp(text, '^no\s+(.+)$', 'tokens');
    if ~isempty(no_match)
        labels = ones(1, 10);
        labels = remove_parts(labels, no_match{1}{1});
        return;
    end

    % ── "Parts of hand except X" = whole hand except X ──
    if contains(text, 'parts of hand except') || contains(text, 'parts of hand without')
        labels = ones(1, 10);
        exc = regexp(text, '(?:except|without)\s+(.+)', 'tokens');
        if ~isempty(exc)
            labels = remove_parts(labels, exc{1}{1});
        end
        return;
    end

    % ── "Except X and segment" (standalone) = whole hand except X ──
    if startsWith(text, 'except')
        labels = ones(1, 10);
        exc = extractAfter(text, 'except');
        labels = remove_parts(labels, exc);
        return;
    end

    % ── Whole hand / all ──
    if contains(text, 'whole hand') || contains(text, 'full hand') || ...
       strcmp(text, 'all') || contains(text, 'all fingers and segments') || ...
       contains(text, 'all fingertips and segments') || ...
       contains(text, 'all fingertips and thumb and segments')
        labels = ones(1, 10);
        exc = regexp(text, '(?:except|without)\s+(.+)', 'tokens');
        if ~isempty(exc)
            labels = remove_parts(labels, exc{1}{1});
        end
        return;
    end

    % ── "And thumb segments" (continuation of previous) ──
    if contains(text, 'and thumb segment')
        % This is a continuation — thumb segment = segment 1, already handled
        % Just mark thumb
        labels(2) = 1;
        return;
    end

    % ── All fingertips (tips only, NOT segments) ──
    if contains(text, 'fingertip') || contains(text, 'finger tip')
        % Tips: thumb(1), index_tip(2), middle_tip(4), ring_tip(6), pinky_tip(8)
        labels([2, 3, 5, 7, 9]) = 1;

        % "and segments" adds all segments
        if contains(text, 'segment')
            labels([4, 6, 8, 10]) = 1;
        end

        % "and palm"
        if contains(text, 'palm')
            labels(1) = 1;
        end

        % "and thumb" (already included in tips but also add for clarity)
        if contains(text, 'and thumb')
            labels(2) = 1;
        end

        % Handle except
        exc = regexp(text, '(?:except|without)\s+(.+)', 'tokens');
        if ~isempty(exc)
            labels = remove_parts(labels, exc{1}{1});
        end
        return;
    end

    % ── Palm (standalone) ──
    if strcmp(text, 'palm')
        labels(1) = 1;
        return;
    end

    % ── Individual numbers ──
    word_map = {'zero',0; 'one',1; 'two',2; 'three',3; 'four',4;
                'five',5; 'six',6; 'seven',7; 'eight',8; 'nine',9};
    words = strsplit(text);
    for k = 1:size(word_map, 1)
        for w = 1:length(words)
            if strcmp(words{w}, word_map{k,1})
                labels(word_map{k,2} + 1) = 1;
            end
        end
    end

    % Digit characters
    nums = regexp(text, '\d', 'match');
    for k = 1:length(nums)
        n = str2double(nums{k});
        if n >= 0 && n <= 9
            labels(n + 1) = 1;
        end
    end

    % Finger names
    if contains(text, 'palm'),   labels(1) = 1; end
    if contains(text, 'thumb'),  labels(2) = 1; end
    if contains(text, 'index'),  labels(3) = 1; labels(4) = 1; end
    if contains(text, 'middle'), labels(5) = 1; labels(6) = 1; end
    if contains(text, 'ring'),   labels(7) = 1; labels(8) = 1; end
    if contains(text, 'pinky'),  labels(9) = 1; labels(10) = 1; end
end


function labels = remove_parts(labels, exc)
    if contains(exc, 'palm'),   labels(1) = 0; end
    if contains(exc, 'thumb'),  labels(2) = 0; end
    if contains(exc, 'index'),  labels(3) = 0; labels(4) = 0; end
    if contains(exc, 'middle'), labels(5) = 0; labels(6) = 0; end
    if contains(exc, 'ring'),   labels(7) = 0; labels(8) = 0; end
    if contains(exc, 'pinky'),  labels(9) = 0; labels(10) = 0; end

    % Number words
    word_map = {'zero',0;'one',1;'two',2;'three',3;'four',4;
                'five',5;'six',6;'seven',7;'eight',8;'nine',9};
    for k = 1:size(word_map, 1)
        if contains(exc, word_map{k,1})
            labels(word_map{k,2} + 1) = 0;
        end
    end

    % Also handle "and segment" in except
    if contains(exc, 'segment')
        if contains(exc, 'thumb'), labels(2) = 0; end
        if contains(exc, 'index'), labels(3) = 0; labels(4) = 0; end
    end
end
