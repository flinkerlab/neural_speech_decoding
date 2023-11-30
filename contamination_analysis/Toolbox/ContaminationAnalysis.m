classdef ContaminationAnalysis
    % Class computing spectrogram correlations between an audio channel and
    % another recording
    %
    % Copyright 2020 INSERM
    % Licence: GPL-3.0-or-later
    % Author(s): Philémon Roussel
    
    properties
        analysis_name
        
        % input paths
        brain_matfile_path
        audio_matfile_path
        
        % output
        results_folder_path
        figures_folder_path
        object_path
        centered_brain_matfile_path
        centered_audio_matfile_path
        artifact_data_path
        audio_spg_path
        brain_spg_path
        
        % input recordings information
        fs
        time
        sample_nb
        channels
        channel_nb
        
        % time selection and artifacts
        select_lgcs
        artifact_lgcs
        analysis_lgcs
        spg_analysis_lgcs
        
        % spectrogram parameters and information
        spg_window_length
        spg_fs
        spg_sample_nb
        spg_time
        spg_freq_bounds
        spg_freqs
        spg_freq_nb
        
        % correlations
        chan_freq_rho
        chan_freq_rho_p
        chan_freq_freq_rho
        chan_freq_freq_rho_p
        
        % cross-correlations
        chan_freq_xcorr
        xcorr_time_lags
        lag_nb
        
        % statistical criterion
        criterion_freq_lims
        criterion_freq_lgcs
        rand_nb
        surrogate_measures
        dataset_measure
        criterion_value
        
    end
    
    %%% PUBLIC METHODS
    methods (Access = public)
        
        function obj = ContaminationAnalysis(...
                results_path,...
                brain_matfile_path,...
                audio_matfile_path,...
                analysis_name)
            
            % Create and store a ContaminationAnalysis object
            %
            % results_path:
            %   path to save the results
            % brain_matfile_path:
            %   brain data matfile path (should respect defined format)
            % brain_matfile_path:
            %   audio data matfile path (should respect defined format)
            % analysis_name (optional):
            %   name of files and figures related to the present analysis
            
            obj.brain_matfile_path = pathify(brain_matfile_path, 'check');
            obj.audio_matfile_path = pathify(audio_matfile_path, 'check');
            
            if ~exist('analysis_name', 'var') || isempty(analysis_name)
                [~, analysis_name] = fileparts(obj.brain_matfile_path);
            end
            
            obj.analysis_name = analysis_name;
            
            % create result folder and store path
            obj.results_folder_path = pathify(...
                strcat(results_path, '/', obj.analysis_name, '/'),...
                'create');
            
            % create figures folder and store path
            obj.figures_folder_path = pathify(...
                [obj.results_folder_path 'figures/'],...
                'create');
            
            % prepare path to save the present object
            obj.object_path = pathify(strcat(...
                obj.results_folder_path, obj.analysis_name, '_object.mat'),...
                'replace');
            
            % load information from brain data file
            brain_mf = matfile(obj.brain_matfile_path);
            brain_data_size = size(brain_mf, 'values');
            obj.sample_nb = brain_data_size(1);
            obj.channel_nb = brain_data_size(2);
            obj.fs = brain_mf.fs;
            
            % check audio data file
            audio_mf = matfile(obj.audio_matfile_path);
            audio_data_size = size(audio_mf, 'values');
            if audio_data_size(1) ~= obj.sample_nb
                error('Inconsistent durations.');
            end
            if audio_mf.fs ~= obj.fs
                error('Inconsistent sampling frequencies.');
            end
            
            % save or create time vector
            if isMatfileField(brain_mf, 'time')
                obj.time = brain_mf.time;
            else
                values_size = size(brain_mf, 'values');
                obj.time = (0:(values_size(1) - 1)) / obj.fs;
            end
            
            % save or create channels table
            if isMatfileField(brain_mf, 'channels')
                obj.channels = brain_mf.channels;
            else
                values_size = size(brain_mf, 'values');
                channels = (1:values_size(2))';
                obj.channels = table(channels, string(channels),...
                    'VariableNames', {'index', 'id'});
            end
            
            % initialization
            obj.select_lgcs = true(obj.sample_nb, 1);
            obj.artifact_lgcs = false(obj.sample_nb, 1);
            obj.analysis_lgcs = true(obj.sample_nb, 1);
            
            saveObject(obj)
        end
        
        function saveObject(obj)
            save(obj.object_path, 'obj')
        end
        
        function obj = loadObject(obj)
            load(obj.object_path, 'obj')
        end
        
        function obj = selectTime(obj,...
                select_periods,...
                exclude_periods)
            
            % Select time samples that will be considered in the analysis
            %
            % select_periods:
            %   2-column array defining start and end times of the time
            %   periods to select.
            % exclude_periods:
            %   2-column array defining start and end times of the time
            %   periods to exclude.
            
            % find selected samples
            if exist('select_periods', 'var') && ~isempty(select_periods)
                sel_lgcs = false(obj.sample_nb, 1);
                
                for p_i = 1:size(select_periods, 1)
                    sel_lgcs(...
                        obj.time >= select_periods(p_i, 1)...
                        & obj.time <= select_periods(p_i, 2)) = true;
                end
            else
                sel_lgcs = true(obj.sample_nb, 1);
            end
            
            % find excluded samples
            exc_lgcs = false(obj.sample_nb, 1);
            if exist('exclude_periods', 'var') && ~isempty(exclude_periods)
                exc_lgcs = false(obj.sample_nb, 1);
                
                for p_i = 1:size(exclude_periods, 1)
                    exc_lgcs(...
                        obj.time >= exclude_periods(p_i, 1)...
                        & obj.time <= exclude_periods(p_i, 2)) = true;
                end
            end
            
            % find finally selected samples
            obj.select_lgcs = sel_lgcs & ~exc_lgcs;
            
            % remove artifacts from selected samples
            obj.analysis_lgcs = obj.select_lgcs & ~obj.artifact_lgcs;
            
            saveObject(obj)
        end
        
        function obj = detectArtifacts(obj,...
                moving_average_span,...
                artifact_threshold_factor,...
                artifact_channel_ratio,...
                artifact_safety_period)
            
            % Detect artifacts occuring on several channels
            %
            % moving_average_span:
            %   Duration (in seconds) of the moving average window that is
            %   used to detrend the data before artifact detection.
            % artifact_threshold_factor:
            %   'artifact_threshold_factor' multiplied by the MAD of a
            %   given channel defines the artifact threshold of this
            %   channel.
            % artifact_channel_ratio:
            %   Ratio of channels crossing their threshold for a sample to
            %   be considered as an artifact
            % artifact_safety_period:
            %   Period of time (in seconds) before and after artifact in
            %   which samples are also considered as artifacts
            
            brain_mf = matfile(obj.brain_matfile_path);
            
            artifact_lgc_mat = false(obj.sample_nb, obj.channel_nb);
            
            obj.centered_brain_matfile_path = pathify(...
                [obj.results_folder_path 'centered_brain_data.mat']);
            centered_brain_mf = createRecordingMatfile(...
                obj.centered_brain_matfile_path, NaN, obj.fs,...
                'channels', obj.channels,...
                'init_dims', [obj.sample_nb, obj.channel_nb]);
            
            safety_sample_nb = round(artifact_safety_period * obj.fs) * 2 + 1;
            safety_filter = ones(1, safety_sample_nb);
            artifact_thresholds = zeros(obj.channel_nb, 1);
            
            wb_h = timeWaitbar('Detect artifacts...');
            
            for chan_i = 1:obj.channel_nb
                
                chan_values = brain_mf.values(:, chan_i);
                
                % compute and save detrended brain data
                detrended_chan_values = chan_values...
                    - applyMovingAverage(chan_values,...
                    moving_average_span, obj.fs, 'extend');
                centered_brain_mf.values(:, chan_i) = detrended_chan_values;
                
                % compute artifact threshold based on MAD
                artifact_thresholds(chan_i) = artifact_threshold_factor...
                    * 1.4826 * median(abs(...
                    detrended_chan_values(obj.select_lgcs) -...
                    median(detrended_chan_values(obj.select_lgcs),...
                    'omitnan')), 'omitnan');
                
                artifact_lgc_mat(obj.select_lgcs, chan_i) =...
                    abs(detrended_chan_values(obj.select_lgcs))...
                    > artifact_thresholds(chan_i);
                
                updateTimeWaitbar(wb_h, chan_i / obj.channel_nb)
            end
            
            % define artifact samples when enough channels cross the
            % threshold
            temp_artifact_lgcs = sum(artifact_lgc_mat, 2)...
                >= obj.channel_nb * artifact_channel_ratio;
            
            % extend artifacts to preceding and following samples and save
            obj.artifact_lgcs = symFilter(safety_filter, temp_artifact_lgcs) >= 1;
            
            % remove artifacts from selected samples
            obj.analysis_lgcs = obj.select_lgcs & ~obj.artifact_lgcs;
            
            obj.artifact_data_path = [obj.results_folder_path 'artifacts_data.mat'];
            save(obj.artifact_data_path, 'artifact_thresholds', 'artifact_lgc_mat')
            
            close(wb_h)
            
            saveObject(obj)
        end
        
        function varargout = displayArtifacts(obj, display_channel_nb)
            
            % Display the results of the artifact detection and save the
            % figure
            %
            % display_channel_nb:
            %   Number of channels to show. The first half of the
            %   displayed channels are the channels with the highest
            %   numbers of artifact samples and the second half are the
            %   ones with the lowest numbers.
            %
            % Can return figure handle.
            
            % prepare data
            MA_removed_mf = matfile(obj.centered_brain_matfile_path);
            artifact_data_mf = matfile(obj.artifact_data_path);
            artifact_lgc_mat = artifact_data_mf.artifact_lgc_mat;
            artifact_thresholds = artifact_data_mf.artifact_thresholds;
            
            % define number of channels for both ratio type
            high_ratio_nb = ceil(display_channel_nb / 2);
            low_ratio_nb = floor(display_channel_nb / 2);
            
            % sort artifact ratios
            artifact_ratios = mean(artifact_lgc_mat(obj.select_lgcs, :));
            [~, chan_idxs] = sort(artifact_ratios, 'descend');
            sel_chan_idxs = [...
                chan_idxs(1:high_ratio_nb)...
                chan_idxs((end - low_ratio_nb + 1):end)];
            
            % create figure
            [fig, new_bool] = setFigure([obj.analysis_name ': Artifacts']);
            obj.m_maximizeOrClear(fig, new_bool)
            
            % preallocate axes handle storage
            axes_h = NaN(display_channel_nb + 1, 1);
            
            % display audio channel
            subplot(display_channel_nb + 1, 1, 1)
            axes_h(1) = gca;
            mf = matfile(obj.audio_matfile_path);
            plot(obj.time, mf.values)
            ylabel('Amplitude')
            title("Audio channel")
            h = legend({'recording artifacts'},...
                'Location', 'eastoutside');
            axis tight
            set(h, 'visible', 'off')
            
            for i = 1:display_channel_nb
                
                % channel ratio type
                if i <= high_ratio_nb
                    ratio_type = 'high';
                else
                    ratio_type = 'low';
                end
                
                % plot channel data
                subplot(display_channel_nb + 1, 1, i + 1)
                axes_h(i + 1) = gca;
                
                vals = MA_removed_mf.values(:, sel_chan_idxs(i));
                plot(obj.time, vals)
                
                hold on
                plot([obj.time(1) obj.time(end)],...
                    [artifact_thresholds(sel_chan_idxs(i))...
                    artifact_thresholds(sel_chan_idxs(i))],...
                    'k--')
                plot([obj.time(1) obj.time(end)],...
                    -[artifact_thresholds(sel_chan_idxs(i))...
                    artifact_thresholds(sel_chan_idxs(i))],...
                    'k--')
                plot(obj.time,...
                    1.5 * artifact_thresholds(sel_chan_idxs(i))...
                    * artifact_lgc_mat(:,sel_chan_idxs(i)))
                plot(obj.time, 2 * artifact_thresholds(sel_chan_idxs(i))...
                    * obj.artifact_lgcs)
                plot(obj.time, 2.5 * artifact_thresholds(sel_chan_idxs(i))...
                    * ~obj.select_lgcs)
                hold off
                
                ylabel('Amplitude')
                
                % style
                axis tight
                drawnow
                legend({'detrended data', '+threshold', '-threshold',...
                    'channel artifacts', 'recording artifacts', 'excluded samples'},...
                    'Location', 'eastoutside')
                chan_ratio = artifact_ratios(sel_chan_idxs(i));
                title(strcat("Channel ",...
                    obj.channels{sel_chan_idxs(i), 'id'}, ": ",...
                    num2str(round(chan_ratio * 100, 2)), "% timepoints crossing threshold",...
                    " (", ratio_type, ")"))
                drawnow
                
            end
            
            % samples selection
            obj = m_selectSpectrogramSamples(obj);
            analysis_sample_nb = nnz(obj.analysis_lgcs);
            analyis_duration = analysis_sample_nb / obj.fs;
            
            % style
            sgtitle({[num2str(round(...
                mean(obj.artifact_lgcs(obj.select_lgcs)) * 100, 2))...
                '% of artifact timepoints'],...
                ['(' num2str(round(analyis_duration)) 's kept for analysis)']})
            xlabel('Time (s)')
            
            % link
            linkaxes(axes_h, 'x')
            
            % save figure
            hgsave(fig, [obj.figures_folder_path 'artifacts.fig'], '-v7.3')
            
            if nargout > 0
                varargout{1} = fig;
            end
            
        end
        
        function obj = computeSpectrograms(obj, window_duration,...
                spg_fs, spg_freq_bounds)
            
            % Compute the spectrograms of the audio and brain recordings
            % The audio signal is centered using moving average before
            % being processed.
            %
            % window_duration:
            %   Duration of the spectrogram window (in seconds).
            % spg_fs:
            %   Desired sampling frequency of the spectrogram.
            % spg_freq_bounds:
            %   2-element vector containing the lowest and the highest
            %   frequencies considered in the spectrogram (if empty, all
            %   frequency bins are kept).
            
            % fixed parameters
            audio_MA_window_duration = 1; % in seconds
            
            % spectrogram parameters
            obj.spg_window_length = round(window_duration * obj.fs);
            obj.spg_fs = spg_fs;
            obj.spg_freq_bounds = spg_freq_bounds;
            padding_method = "symmetrize";
            
            % copy audio file (to keep channel info. etc.)
            obj.centered_audio_matfile_path = pathify([obj.results_folder_path...
                'centered_audio_data.mat']);
            status = copyfile(obj.audio_matfile_path,...
                obj.centered_audio_matfile_path);
            if ~status
                error('Audio file could not be copied.')
            end
            
            % remove moving average
            audio_values = getMatfileVar(obj.audio_matfile_path, 'values');
            detrended_audio_values = audio_values - applyMovingAverage(...
                audio_values, audio_MA_window_duration, obj.fs, 'extend');
            
            % store result (size should not change)
            setMatfileData(obj.centered_audio_matfile_path, 'values',...
                detrended_audio_values, {':', ':'});
            
            % audio spectrogram
            obj.audio_spg_path = [obj.results_folder_path 'audio_spg.mat'];
            [~, spg_t, freqs] =...
                computeMultichannelSpectrogram(...
                obj.centered_audio_matfile_path, obj.spg_fs,...
                'window_duration', window_duration,...
                'padding_method', padding_method,...
                'frequency_limits', obj.spg_freq_bounds,...
                'output_path', obj.audio_spg_path);
            
            % store information in properties
            obj.spg_time = spg_t;
            obj.spg_freqs = freqs;
            obj.spg_freq_nb = length(obj.spg_freqs);
            obj.spg_sample_nb = length(spg_t);
            
            % brain spectrogram
            obj.brain_spg_path =...
                [obj.results_folder_path 'multichannel_spg_data.mat'];
            computeMultichannelSpectrogram(...
                obj.brain_matfile_path, obj.spg_fs,...
                'window_duration', window_duration,...
                'padding_method', padding_method,...
                'frequency_limits', obj.spg_freq_bounds,...
                'output_path', obj.brain_spg_path);
            
            saveObject(obj)
        end
        
        function obj = computeSpectrogramCorrelations(obj)
            
            % Compute correlations between the spectrogram of the audio and
            % the spectrograms of the different channels
            
            %%% SAMPLES SELECTION
            
            obj = m_selectSpectrogramSamples(obj);
            
            %%% CORRELATION COMPUTATION
            
            brain_spg_mf = matfile(obj.brain_spg_path);
            audio_spg_mf = matfile(obj.audio_spg_path);
            audio_pspg = getMatfileData(...
                audio_spg_mf, 'values', {obj.spg_analysis_lgcs, ':'});
            
            % preallocate
            obj.chan_freq_rho = NaN(obj.channel_nb, obj.spg_freq_nb);
            obj.chan_freq_rho_p = NaN(obj.channel_nb, obj.spg_freq_nb);
            obj.chan_freq_freq_rho = NaN(obj.channel_nb, obj.spg_freq_nb,...
                obj.spg_freq_nb);
            obj.chan_freq_freq_rho_p = NaN(obj.channel_nb, obj.spg_freq_nb,...
                obj.spg_freq_nb);
            
            % compute correlations and p-values
            wb_h = timeWaitbar('Computing correlation');
            
            for chan_i = 1:obj.channel_nb
                
                brain_pspg = squeeze(getMatfileData(...
                    brain_spg_mf, 'values', {obj.spg_analysis_lgcs, ':', chan_i}));
                
                [r, p] = corr(audio_pspg, brain_pspg);
                
                obj.chan_freq_freq_rho(chan_i, :, :) = r;
                obj.chan_freq_freq_rho_p(chan_i, :, :) = p;
                
                obj.chan_freq_rho(chan_i, :, :) = diag(r);
                obj.chan_freq_rho_p(chan_i, :, :) = diag(p);
                
                updateTimeWaitbar(wb_h, chan_i / obj.channel_nb)
                
            end
            
            close(wb_h)
            
            saveObject(obj)
        end
        
        function varargout = displayCorrelations(obj, disp_freqs_bounds,...
                display_channels, colormap_limits)
            
            % Display the spectrogram correlations and save the figures.
            %
            % disp_freqs_bounds:
            %   2-element vector containing the lowest and the highest
            %   frequencies displayed in the spectrogram (if empty, all
            %   frequency bins are kept).
            % display_channels:
            %   'index' or 'id' of the channels to be displayed.
            % colormap_limits:
            %   2-element vector containing the lowest and the limits of
            %   the colormap displaying the z-scored spectrograls.
            %
            % Can return figure handles.
            
            % fixed parameters
            test_alpha = 0.05;
            display_type = 'imagesc';
            
            % default parameter values
            if ~exist('disp_freqs_bounds', 'var') || isempty(disp_freqs_bounds)
                disp_freqs_bounds = obj.spg_freq_bounds;
            end
            
            if ~exist('colormap_limits', 'var') || isempty(colormap_limits)
                colormap_limits = [0 5];
            end
            
            % process frequency limits
            disp_freq_lgcs = (obj.spg_freqs >= disp_freqs_bounds(1)...
                & obj.spg_freqs <= disp_freqs_bounds(2));
            disp_freqs = obj.spg_freqs(disp_freq_lgcs);
            
            % load data
            audio_spg_mf = matfile(obj.audio_spg_path);
            brain_spg_mf = matfile(obj.brain_spg_path);
            audio_spg = getMatfileData(audio_spg_mf, 'values', {':', disp_freq_lgcs});
            
            % find significant audio-neural correlations (Bonferroni correction)
            signif_corr_lgcs = obj.chan_freq_rho_p < (test_alpha / numel(obj.chan_freq_rho_p));
            signif_chan_freq_rho = obj.chan_freq_rho;
            signif_chan_freq_rho(~signif_corr_lgcs) = NaN;
            
            % find most correlated channels
            chan_max_rho = max(signif_chan_freq_rho(:, disp_freq_lgcs), [], 2, 'omitnan');
            chan_max_rho(isnan(chan_max_rho)) = -Inf;
            [~, max_rho_chan_idxs] = sort(chan_max_rho, 'descend');
            
            % define channels to display
            if ~exist('display_channels', 'var') || isempty(display_channels)
                display_channels = obj.channels{max_rho_chan_idxs(1:3), 'id'};
            end
            
            if isnumeric(display_channels)
                [~, display_idxs] = ismember(display_channels, obj.channels.index);
            else
                [~, display_idxs] = ismember(display_channels, obj.channels.id);
            end
            
            display_channels = obj.channels{display_idxs, 'id'};
            
            display_chan_nb = length(display_channels);
            
            % average speech spectrum
            speech_spectrum = mean(audio_spg(obj.spg_analysis_lgcs, :));
            
            %%% Figure 'most correlated channels'
            [fig_corr_chans, new_bool] = setFigure([obj.analysis_name ': Most correlated channels']);
            obj.m_maximizeOrClear(fig_corr_chans, new_bool)
            spg_axes_handles = [];
            corr_axes_handles = [];
            imagesc_axes_handles = [];
            col_nb = 5;
            spg_col_nb = 3;
            
            % audio spectrogram
            subplot(display_chan_nb + 1, col_nb, 1:spg_col_nb)
            data = audio_spg;
            data(~obj.spg_analysis_lgcs, :) = NaN;
            displaySpectrogram(normStd((data + eps)', 2), obj.spg_time, disp_freqs, [], display_type)
            spg_axes_handles = [spg_axes_handles ; gca];
            
            % style
            colormap(gca, viridis())
            cb_h = colorbar('TickDirection', 'out');
            cb_h.Label.String = 'z-scored power';
            caxis(colormap_limits)
            ylabel('Frequency (Hz)')
            title('Audio spectrogram')
            set(gca, 'TickDir', 'out')
            box off
            
            for c = 1:display_chan_nb
                
                chan_i = display_idxs(c);
                
                % channel spectrogram
                h = subplot(display_chan_nb + 1, col_nb, (c * col_nb) + (1:spg_col_nb));
                data = squeeze(getMatfileData(brain_spg_mf, 'values', {':', disp_freq_lgcs, chan_i}));
                data(~obj.spg_analysis_lgcs, :) = NaN;
                displaySpectrogram(normStd((data + eps)', 2), obj.spg_time, disp_freqs, [], display_type);
                spg_axes_handles = [spg_axes_handles ; h];
                
                % style
                colormap(h, viridis())
                cb_h = colorbar('TickDirection', 'out');
                cb_h.Label.String = 'Z-scored power';
                caxis(colormap_limits)
                title(strcat("Channel ", display_channels(c), " spectrogram"));
                ylabel('Frequency (Hz)')
                set(gca, 'TickDir', 'out')
                box off
                
                % correlations barplot
                h = subplot(display_chan_nb + 1, col_nb, (c * col_nb) + spg_col_nb + 1);
                bar(disp_freqs,...
                    signif_chan_freq_rho(chan_i, disp_freq_lgcs),...
                    'horizontal', 'on', 'barwidth', 1);
                corr_axes_handles = [corr_axes_handles ; h];
                
                % style
                ylabel('Frequency (Hz)')
                set(gca, 'TickDir', 'out')
                box off
                drawnow
                
                % correlations matrices
                h = subplot(display_chan_nb + 1, col_nb, (c * col_nb) + spg_col_nb + 2);
                corr_vals = squeeze(obj.chan_freq_freq_rho(chan_i,...
                    disp_freq_lgcs, disp_freq_lgcs));
                
                imagesc(disp_freqs, disp_freqs, corr_vals)
                
                max_val = max(max(corr_vals, [], 1), [], 2);
                caxis([0 max_val])
                
                hold on
                
                plot(disp_freqs, disp_freqs - 75, 'w')
                plot(disp_freqs, disp_freqs + 75, 'w')
                
                hold off
                
                imagesc_axes_handles = [imagesc_axes_handles ; h];
                
                % style
                title('Correlation (r)')
                colormap(h, inferno())
                colorbar('TickDirection', 'out')
                set(h, 'YDir', 'normal', 'TickDir', 'out', 'box', 'off')
                pbaspect([1 1 1])
                xlabel('Recording freq. (Hz)')
                ylabel('Audio freq. (Hz)')
                
            end
            
            clear data
            
            % style
            axes(spg_axes_handles(end))
            xlabel('Time (s)')
            axes(corr_axes_handles(end));
            xlabel('Correlation (r)')
            
            % link
            linkaxes(spg_axes_handles, 'xy')
            linkaxes(corr_axes_handles, 'xy')
            linkaxes(imagesc_axes_handles, 'xy')
            
            %%% Figure 'overview'
            
            [fig_overview, new_bool] = setFigure([obj.analysis_name ': Correlations overview']);
            obj.m_maximizeOrClear(fig_overview, new_bool)
            
            drawnow % avoid axis late resize problems
            
            left_color = [0 0 0];
            right_color = [1 0 0];
            set(fig_overview, 'defaultAxesColorOrder', [left_color; right_color]);
            
            % correlations + spectrum subplot
            s(1) = subplot(2, 5, 1:3);
            title('All channels correlations and audio spectrum')
            
            % correlations
            yyaxis left
            plot(disp_freqs, obj.chan_freq_rho(:, disp_freq_lgcs)', '-',...
                'Color', [0 0 0 max(0.15, 10 / obj.channel_nb)])
            
            % style
            ylabel('Correlation (r)')
            min_val = squeeze(min(min(obj.chan_freq_rho, [], 'omitnan'),...
                [], 'omitnan'));
            ylim([min_val-0.1 1.1])
            y_lims = ylim;
            xlabel('Frequency (Hz)')
            
            % spectrum
            yyaxis right
            plot(disp_freqs, speech_spectrum / max(speech_spectrum), 'r',...
                'LineWidth', 1.5)
            
            % style
            ylabel('Audio spectrum (a.u.)')
            xlim([min(disp_freqs) max(disp_freqs)])
            ylim(y_lims)
            set(gca, 'TickDir', 'out', 'box', 'off')
            
            % heatmap subplot
            s(2) = subplot(2, 5, 6:8);
            displaySpectrogram(signif_chan_freq_rho(:, disp_freq_lgcs),...
                disp_freqs, 1:obj.channel_nb)
            title('Significant correlations (Bonferroni correction)')
            
            % style
            colormap(s(2), inferno())
            cb_h = colorbar(s(2), 'TickDirection', 'out');
            cb_h.Label.String = 'Correlation (r)';
            yticks(1:obj.channel_nb)
            yticklabels(obj.channels.id)
            ylabel('Channels')
            xlabel('Frequency (Hz)')
            set(gca, 'TickDir', 'out')
            box off
            
            % align and link the 2 subplots
            drawnow
            pos = get(s, 'position');
            pos{1}(3) = pos{2}(3);
            set(s(1),'position',pos{1})
            s(1).ActivePositionProperty = 'position';
            s(2).ActivePositionProperty = 'position';
            linkaxes([s(1) s(2)], 'x')
            
            % contamination matrix heat map
            s(3) = subplot(2, 5, [4:5 9:10]);
            max_freq_freq_rho = squeeze(max(obj.chan_freq_freq_rho, [],...
                1, 'omitnan'));
            displaySpectrogram(max_freq_freq_rho, disp_freqs, disp_freqs)

            hold on
            
            plot(disp_freqs, disp_freqs - 75, 'w')
            plot(disp_freqs, disp_freqs + 75, 'w')
            
            hold off
            
            colormap(s(3), inferno())
            max_val = max(max(max_freq_freq_rho, [], 1), [], 2);
            caxis([0, max_val])
            h = colorbar('TickDirection', 'out');
            set(get(h,'label'), 'string', 'Correlation (r)')
            set(s(3), 'YDir', 'normal', 'TickDir', 'out', 'box', 'off')
            pbaspect([1 1 1])
            title( 'Contamination matrix')
            
            ylabel('Audio frequency (Hz)')
            xlabel('Recording frequency (Hz)')
            
            % save figures
            hgsave(fig_corr_chans,...
                [obj.figures_folder_path 'most_correlated_channels.fig'],...
                '-v7.3')
            hgsave(fig_overview,...
                [obj.figures_folder_path 'correlations_overview.fig'],...
                '-v7.3')
            
            % output
            if nargout > 0
                varargout{1} = fig_corr_chans;
            end
            if nargout > 1
                varargout{2} = fig_overview;
            end
        end
        
        function obj = computeSpectrogramCrossCorrelations(obj, max_time_lag)
            
            % Compute cross-correlations between the spectrogram of the audio and
            % the spectrograms of the different channels.
            %
            % max_time_lag:
            %   Maximum absolute time lag in seconds considered when applying positive
            %   and negative delays to the audio spectrogram.
            
            max_sample_lag = round(max_time_lag * obj.spg_fs);
            obj.xcorr_time_lags = ((-max_sample_lag):max_sample_lag)...
                / obj.spg_fs;
            obj.lag_nb = length(obj.xcorr_time_lags);
            
            %%% SAMPLES SELECTION
            
            obj = m_selectSpectrogramSamples(obj);
            
            %%% CROSS-CORRELATION COMPUTATION
            
            brain_spg_mf = matfile(obj.brain_spg_path);
            audio_spg_mf = matfile(obj.audio_spg_path);
            audio_pspg = getMatfileData(...
                audio_spg_mf, 'values', {obj.spg_analysis_lgcs, ':'});
            norm_audio_pspg = normStd(audio_pspg);
            
            clear audio_pspg
            
            % preallocate
            obj.chan_freq_xcorr = NaN(obj.channel_nb, obj.spg_freq_nb,...
                obj.lag_nb);
            
            % compute correlations and p-values
            wb_h = timeWaitbar('Computing cross-correlation');
            
            for chan_i = 1:obj.channel_nb
                
                brain_pspg = squeeze(getMatfileData(...
                    brain_spg_mf, 'values',...
                    {obj.spg_analysis_lgcs, ':', chan_i}));
                
                norm_brain_pspg = normStd(brain_pspg);
                
                for f_i = 1:obj.spg_freq_nb
                    
                    obj.chan_freq_xcorr(chan_i, f_i, :) = xcorr(...
                        norm_audio_pspg(:, f_i),...
                        norm_brain_pspg(:, f_i),...
                        max_sample_lag, 'biased');
                    
                    updateTimeWaitbar(wb_h,...
                        ((chan_i - 1) * obj.spg_freq_nb + f_i)...
                        / (obj.channel_nb * obj.spg_freq_nb))
                    
                end
            end
            
            close(wb_h)
            
            saveObject(obj)
        end
        
        function varargout = displayCrossCorrelations(obj, min_max_freqs,...
                top_corr_ratio)
            
            % Display cross-correlations between the audio and recording
            % channels spectrograms.
            %
            % crosscorr_min_max_freqs:
            %   2-element vector containing the lowest and the highest
            %   frequencies to be considered (if empty, all frequency bins
            %   are kept).
            % top_corr_ratio:
            %   Ratio of the highest cross-correlograms to display. 0.01
            %   means that the 1% of cross-correlograms reaching the
            %   highest values will be displayed.
            
            if ~exist('min_max_freqs', 'var') || isempty(min_max_freqs)
                min_max_freqs = [0 Inf];
            end
            
            if ~exist('top_corr_ratio', 'var') || isempty(top_corr_ratio)
                top_corr_ratio = 0.01;
            end
            
            % Flatten matrix
            % Channel x Frequency x Lag to (Channel-Frequency) x Lag
            flat_xcorr = reshape(obj.chan_freq_xcorr,...
                obj.channel_nb * obj.spg_freq_nb, obj.lag_nb)';
            
            % Get max. correlation for each channel-frequency pair
            [max_vals, max_idxs] = max(flat_xcorr);
            
            % Keep 'top_corr_ratio' of the best values
            [~, sorted_idxs] = sort(max_vals, 'descend');
            n = round(numel(max_vals) * top_corr_ratio);
            sel_idxs = sorted_idxs(1:n);
            max_idxs = max_idxs(sel_idxs);
            flat_xcorr = flat_xcorr(:, sel_idxs);
            
            % Prepare figure
            [fig, new_bool] = setFigure(...
                [obj.analysis_name ': Cross-correlations']);
            obj.m_maximizeOrClear(fig, new_bool)
            
            % curves plot
            subplot(211)
            
            plot(obj.xcorr_time_lags', flat_xcorr', '-', 'Color',...
                [0 0 0 max(0.1, 10 / size(flat_xcorr, 1))])
            
            % style
            min_lag = min(obj.xcorr_time_lags);
            max_lag = max(obj.xcorr_time_lags);
            xlim([min_lag max_lag])
            ylabel('Correlation (r)')
            
            % histogram plot
            subplot(212)
            
            bin_edges = [obj.xcorr_time_lags - (0.5 / obj.spg_fs)...
                obj.xcorr_time_lags(end) + (0.5 / obj.spg_fs)];
            histogram(obj.xcorr_time_lags(max_idxs),...
                'BinEdges', bin_edges, 'Normalization', 'count')
            
            % style
            xlim([min_lag max_lag])
            xlim([min_lag max_lag])
            xlabel('Time lag (s)')
            ylabel('Count')
            
            linkFigureAxes('x')
            
            sgtitle({['Audio-brain cross-correlations ('...
                num2str(top_corr_ratio) ' top values ratio in '...
                num2str(min_max_freqs(1)) '-' num2str(min_max_freqs(2)) 'Hz)'],...
                '(positive time lag means delaying audio maximizes correlation)'})
            
            % save figure
            hgsave(fig,...
                [obj.figures_folder_path 'cross-correlations.fig'],...
                '-v7.3')
            
            % output
            if nargout > 0
                varargout{1} = fig_corr_chans;
            end
            
        end
        
        function [obj, criterion_value] = computeStatisticalCriterion(obj, criterion_freq_lims, rand_nb)
            
            % Compute statistical criterion P.
            %
            % criterion_min_max_freqs:
            %   2-element vector containing the lowest and the highest
            %   frequencies to be considered (if empty, all frequency bins are kept).
            
            if ~exist('criterion_freq_lims', 'var') || isempty(criterion_freq_lims)
                criterion_freq_lims = [obj.spg_freqs(1) obj.spg_freqs(end)];
                
            elseif numel(criterion_freq_lims) < 2
                criterion_freq_lims(2) = obj.spg_freqs(end);
                
            end
            
            if ~exist('rand_nb', 'var') || isempty(rand_nb)
                obj.rand_nb = 1e4;
            else
                obj.rand_nb = rand_nb;
            end
            
            % frequency selection
            obj.criterion_freq_lims = criterion_freq_lims;
            obj.criterion_freq_lgcs = obj.spg_freqs >= obj.criterion_freq_lims(1)...
                & obj.spg_freqs <= obj.criterion_freq_lims(2);
            
            audio_neur_corr_mat = squeeze(max(obj.chan_freq_freq_rho(: ,...
                obj.criterion_freq_lgcs, obj.criterion_freq_lgcs)));
            
            % dataset measure
            diag_values = diag(audio_neur_corr_mat);
            obj.dataset_measure = mean(diag_values);
            
            % surrogate measures
            obj.surrogate_measures = NaN(1, obj.rand_nb);
            
            rand_col_row = randi(2, obj.rand_nb, 1);
            
            for i = 1:obj.rand_nb
                
                rand_elements = shuffleArray(audio_neur_corr_mat, rand_col_row(i));
                
                obj.surrogate_measures(i) = mean(...
                    diag(rand_elements));
                
            end
            
            % criterion
            criterion_value = mean(obj.surrogate_measures >= obj.dataset_measure);
            obj.criterion_value = criterion_value;
            
            saveObject(obj);
            
        end
        
        function displayStatisticalCriterion(obj)
            
            % Display statistical criterion P
            
            % compute contamination matrix
            audio_neur_corr_mat = squeeze(max(obj.chan_freq_freq_rho));
            
            % select frequencies
            audio_neur_corr_mat(~obj.criterion_freq_lgcs, :) = NaN;
            audio_neur_corr_mat(:, ~obj.criterion_freq_lgcs) = NaN;
            
            % prepare figure
            [fig, new_bool] = setFigure(...
                [obj.analysis_name ': Statistical criterion']);
            obj.m_maximizeOrClear(fig, new_bool)
            
            % contamination matrix
            subplot(1, 2, 1)
            
            displaySpectrogram(audio_neur_corr_mat, obj.spg_freqs,...
                obj.spg_freqs)
            
            colormap(gca, inferno())
            c_h = colorbar('TickDirection', 'out');
            set(get(c_h, 'label'), 'string', 'Correlation (r)')
            pbaspect([1 1 1])
            set(gca, 'TickDir', 'out', 'box', 'off')
            
            hold on
            
            plot(obj.spg_freqs, obj.spg_freqs + 75, 'w')
            plot(obj.spg_freqs, obj.spg_freqs - 75, 'w')
            
            hold off
            
            xlabel('Recording frequency')
            ylabel('Audio frequency')
            title({'Contamination matrix',...
                ['Mean diagonal value: ' sprintf('%.1e', obj.dataset_measure)]})
            
            % histogram
            subplot(1, 2, 2)
            
            histogram(obj.surrogate_measures, round(sqrt(obj.rand_nb)),...
                'Normalization', 'count', 'EdgeColor', 'none',...
                'FaceColor', 'k', 'FaceAlpha', 1)
            
            hold on
            
            lims = axis;
            line(obj.dataset_measure * [1 1], lims(3:4),...
                'Color', 'r', 'LineWidth', 2);
            
            hold off
            
            xlabel('Mean diagonal value')
            ylabel('Count')
            title({'Statistical criterion',...
                ['P = ' sprintf('%.1e', obj.criterion_value)]})
            set(gca, 'TickDir', 'out', 'box', 'off')
            
            % save figure
            hgsave(fig, [obj.figures_folder_path 'criterion.fig'], '-v7.3')
            
        end
        
    end
    
    %%% PRIVATE METHODS
    methods (Access = private)
        
        function obj = m_selectSpectrogramSamples(obj)
            
            % Internal function used to select spectrogram samples for
            % analyses. It excludes any spectrogram sample that has been
            % computed using a raw signal sample that is not selected in
            % obj.analysis_lgcs. The result is stored in
            % obj.spg_analysis_lgcs.
            
            window_duration = obj.spg_window_length / obj.fs;
            window_starts = obj.spg_time - window_duration / 2;
            window_ends = obj.spg_time + window_duration / 2;
            
            obj.spg_analysis_lgcs = true(obj.spg_sample_nb, 1);
            
            % find first and last indexes of each spectrogram window
            window_start_indexes = searchSortedVector(obj.time,...
                window_starts, 'supeg');
            window_end_indexes = searchSortedVector(obj.time,...
                window_ends, 'infeg');
            
            % find spectrogram samples computed on windows including
            % non-selected samples to remove them from further analyses
            for i = 1:obj.spg_sample_nb
                
                obj.spg_analysis_lgcs(i) = ~any(~obj.analysis_lgcs(...
                    window_start_indexes{i}:window_end_indexes{i}));
                
            end
        end
        
        function m_maximizeOrClear(~, fig_handle, new_bool)
            
            % Internal function that is used when figures are created or
            % updated. If 'new_bool' is true, the figure window is
            % maximized. If it is false, the content of the figure window
            % is cleared but the size is not modified. Thus, if the figure
            % has been manually resized, its size is preserved when it is
            % updated.
            
            if new_bool
                set(fig_handle, 'WindowState', 'Maximized')
            else
                clf
            end
        end
    end
end
