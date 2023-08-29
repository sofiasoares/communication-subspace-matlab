clear all
SET_CONSTS

% Vector containing the interaction dimensionalities to use when fitting
% RRR. 0 predictive dimensions results in using the mean for prediction.
numDimsUsedForPrediction = 1:20;
% Number of cross validation folds.
cvNumFolds = 10;
fit_thresh = 0.01;
maxNeuTrgt = 35;
qFA = 0:maxNeuTrgt-1;
saveFig = 0;
maxNeuSrc = 80;
dim_fract = 0.8;%fraction of full performance to use in estimating dimensionality of subspace
useParallel = 0;
dimStruct = struct;
dataFolder = 'Z:\HarveyLab\Tier1\Sofia\Data\CommunicationSubspace\mat_sample';
animalList = dir(dataFolder);
% Filter out non-folders
animalList = animalList([animalList.isdir]);  % Keep only items that are directories
animalList = {animalList.name};  % Store the names in a cell array
animalList = animalList(~ismember(animalList, {'.', '..'}));
animalList = animalList(contains(animalList, 'SS'));
sumStruct = struct;
withinDayStruct = struct;

%%
for iMouse = 1:numel(animalList)
    dimStruct = struct;
    thisMouse = animalList{iMouse};
    sessionsFolder = fullfile(dataFolder,thisMouse);
    sessionList = dir(sessionsFolder);
    % Filter out non-folders
    sessionList = sessionList([sessionList.isdir]);  % Keep only items that are directories
    sessionList = {sessionList.name};  % Store the names in a cell array
    sessionList = sessionList(~ismember(sessionList, {'.', '..'}));
    
    for iSession = 1:numel(sessionList)
        sessionLabel = sessionList{iSession};
        disp(['Working on animal ' thisMouse ' and session ' sessionLabel]);
        thisSessionPath = fullfile(sessionsFolder,sessionLabel);
        
        pathNameFigs = fullfile(thisSessionPath,'figs');  % Specify the folder name
        
        if ~exist(pathNameFigs, 'dir')
            mkdir(pathNameFigs);
            disp('Folder created successfully.');
        else
            disp('Folder already exists.');
        end
        
        files = dir(fullfile(thisSessionPath, '*.mat'));
        thisSessionData = struct;
        % Find the index of the underscore
        underscoreIndex = strfind(thisSessionPath, '_');
        slashIndex = strfind(thisSessionPath, '\');
        % Extract the two substrings for the sessions, before and after the underscore
        sessionStr = thisSessionPath(slashIndex(end)+1:underscoreIndex(end)-1);
        sessionTestStr = thisSessionPath(underscoreIndex(end)+1:end);
        
        % Loop through each file
        for i = 1:numel(files)
            filePath = fullfile(thisSessionPath, files(i).name);
            tmp = load(filePath);
            varName = fieldnames(tmp);
            varName = varName{1};
            thisSessionData.(varName) = tmp.(varName);
        end
        
        this_session_areas_full = fieldnames(thisSessionData);
        this_session_areas = this_session_areas_full(~contains(this_session_areas_full,'test'));
        
        for iAreaSource = 1:numel(this_session_areas)
            
            thisSourceArea = this_session_areas{iAreaSource};
            disp(['Working on source area ' thisSourceArea]);
            
            data2use = thisSessionData.(thisSourceArea);
            
            if size(data2use,1)>=[maxNeuTrgt+maxNeuSrc]
                  
                randI = randperm(size(data2use,1));
                selectedI_source = sort(randI(1:maxNeuTrgt));
                keepIndices = true(size(data2use,1), 1);
                keepIndices(selectedI_source) = false;
                iKeep = find(keepIndices==1);
                iKeepRand = randperm(size(iKeep,1));
                iKeepRand_source = iKeep(sort(iKeepRand(1:maxNeuSrc)));
                
                
                X_source = data2use(iKeepRand_source,:)';
                Y_target_same = data2use(selectedI_source,:)';
                
                for iAreaTarget = 1:numel(this_session_areas)
                    %SELECT AGAIN THE DATA FROM THE SOURCE AREA FROM THE
                    %FIRST DAY (because bellow we loop for the test day)
                    thisSourceArea = this_session_areas{iAreaSource};
                    data2use = thisSessionData.(thisSourceArea);
                    X_source = data2use(iKeepRand_source,:)';
                    Y_target_same = data2use(selectedI_source,:)';
                    
                    thisTargetArea = this_session_areas{iAreaTarget};
                    data2use = thisSessionData.(thisTargetArea);
                    
                    if size(data2use,1)>=maxNeuTrgt
                        
                        if strcmp(thisTargetArea,thisSourceArea)==0
                            randI = randperm(size(data2use,1));
                            selectedI_target = sort(randI(1:maxNeuTrgt));
                            
                            
                            Y_target = data2use(selectedI_target,:)';
                            
                        elseif strcmp(thisTargetArea,thisSourceArea)==1
                            
                            Y_target = Y_target_same;
                        end
                        
                        for iTest = 1:2
                            
                            if iTest==2
                                thisSourceArea = [thisSourceArea '_test'];
                                thisTargetArea = [thisTargetArea '_test'];
                                data2use = thisSessionData.(thisSourceArea);
                                %use same idx as the paired session
                                X_source = data2use(iKeepRand_source,:)';
                                Y_target_same = data2use(selectedI_source,:)';
                                data2use = thisSessionData.(thisTargetArea);
                                
                                if strcmp(thisTargetArea,thisSourceArea)==0
                                    Y_target = data2use(selectedI_target,:)';
                                    
                                elseif strcmp(thisTargetArea,thisSourceArea)==1
                                    
                                    Y_target = Y_target_same;
                                end
                                
                            end
                            
                            figure
                            
                            
                            % Regression cross-validation examples +++++++++++++++++++++++++++++++++++
                            % ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            
                            % Cross-validate Reduced Rank Regression
                            
                            % Initialize default options for cross-validation.
                            cvOptions = statset('crossval');
                            
                            % If the MATLAB parallel toolbox is available, uncomment this line to
                            % enable parallel cross-validation.
                            if useParallel
                                cvOptions.UseParallel = true;
                            end
                            
                            % Regression method to be used.
                            regressMethod = @ReducedRankRegress;
                            
                            % Auxiliary function to be used within the cross-validation routine (type
                            % 'help crossval' for more information). Briefly, it takes as input the
                            % the train and test sets, fits the model to the train set and uses it to
                            % predict the test set, reporting the model's test performance. Here we
                            % use NSE (Normalized Squared Error) as the performance metric. MSE (Mean
                            % Squared Error) is also available.
                            cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
                                (regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
                                numDimsUsedForPrediction, 'LossMeasure', 'NSE');
                            
                            % Cross-validation routine.
                            cvl = crossval(cvFun, Y_target, X_source, ...
                                'KFold', cvNumFolds, ...
                                'Options', cvOptions);
                            
                            % Stores cross-validation results: mean loss and standard error of the
                            % mean across folds.
                            cvLoss = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];
                            
                            % To compute the optimal dimensionality for the regression model, call
                            % ModelSelect:
                            optDimReducedRankRegress = ModelSelect...
                                (cvLoss, numDimsUsedForPrediction);
                            
                            [B, B_, V] = ReducedRankRegress(Y_target, X_source, optDimReducedRankRegress);
                            
                            % Plot Reduced Rank Regression cross-validation results
                            x = numDimsUsedForPrediction;
                            y = 1-cvLoss(1,:);
                            e = cvLoss(2,:);
                            
                            errorbar(x, y, e, 'o--', 'Color', COLOR(Area2,:), ...
                                'MarkerFaceColor', COLOR(Area2,:), 'MarkerSize', 10)
                            
                            xlabel('Number of predictive dimensions')
                            ylabel('Predictive performance')
                            %
                            % Cross-validated Ridge Regression
                            
                            regressMethod = @RidgeRegress;
                            
                            % For Ridge regression, the correct range for lambda can be determined
                            % using:
                            
                            dMaxShrink = .5:.01:1;
                            lambda = GetRidgeLambda(dMaxShrink, X_source);
                            
                            cvParameter = lambda;
                            lossMeasure = 'NSE'; % NSE stands for Normalized Squared Error
                            
                            
                            % Whenever the regression function has extra (potentially optional)
                            % arguments, they are passed to the auxiliary cross-validation function as
                            % name argument pairs.
                            
                            % Ridge Regression has no extra arguments, so the auxiliary
                            % cross-validation function becomes:
                            
                            cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
                                (@RidgeRegress, Ytrain, Xtrain, Ytest, Xtest, lambda, ...
                                'LossMeasure', 'NSE');
                            cvOptions = statset('crossval');
                            
                            cvl = crossval(cvFun, Y_target, X_source, ...
                                'KFold', cvNumFolds, ...
                                'Options', cvOptions);
                            
                            cvLoss = ...
                                [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];
                            
                            [optLambda,optLoss] = ModelSelect...
                                (cvLoss, lambda);
                            %
                            % Plot Ridge Regression cross-validation results
                            x_full = 0;
                            y_full = 1-optLoss(1,:);
                            
                            hold on
                            plot(x_full, y_full, 'o--', 'Color', COLOR(Area2,:), ...
                                'MarkerFaceColor', 'w', 'MarkerSize', 10)
                            hold off
                            
                            xlim([-0.6 numDimsUsedForPrediction(end)+0.6])
                            
                            % Calculate dimensionality of subspace
                            
                            thr = y_full*dim_fract;
                            this_dim = find(y>=thr,1);
                            %                             this_dim = 0;
                            
                            if isempty(this_dim)
                                title([thisSourceArea ' - ' thisTargetArea ' and we could not determine dimensionality - paper would assume ' num2str(optDimReducedRankRegress) ' dimensions'])
                            else
                                
                                title([thisSourceArea ' - ' thisTargetArea ' has ' num2str(this_dim) ' dimensions - paper would assume ' num2str(optDimReducedRankRegress) ' dimensions'])
                            end
                            
                            if saveFig
                                saveas(gcf,fullfile(pathNameFigs, [thisSourceArea '_' thisTargetArea '_cs.pdf']))
                            end
                            sessionField = ['Dates_' strrep(sessionLabel, '-', '')];
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).optDim=optDimReducedRankRegress;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).myDim = this_dim;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).fullPerf = y_full;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).optDimPerf = y(optDimReducedRankRegress);
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).B = B;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).B_ = B_;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).V = V;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).X_source = X_source;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).Y_target = Y_target;
                            
                            
                            %Now do Factor analysis to estimate target area dimensionality
                            cvOptions = statset('crossval');

                            % In order to apply Factor Regression, we must first determine the optimal
                            % dimensionality for the Factor Analysis Model. Here we will ask what is
                            % the dimensionality of the target population

                            qOptDimTargetFA = FactorAnalysisModelSelect( ...
                                CrossValFa(Y_target, qFA, cvNumFolds, cvOptions), qFA);
                            
                            
                            if iTest==1
                                sessionFieldWithinDay = ['Date_' strrep(sessionStr, '-', '_')];
                                withinDayStruct.(thisMouse).(thisSourceArea).(thisTargetArea).(sessionFieldWithinDay).optDimPerf = y(optDimReducedRankRegress);
                                withinDayStruct.(thisMouse).(thisSourceArea).(thisTargetArea).(sessionFieldWithinDay).optDim = optDimReducedRankRegress;
                                withinDayStruct.(thisMouse).(thisSourceArea).(thisTargetArea).(sessionFieldWithinDay).fullPerf = y_full;
                                withinDayStruct.(thisMouse).(thisSourceArea).(thisTargetArea).(sessionFieldWithinDay).targetDimFA = qOptDimTargetFA;

                                
                            elseif iTest==2
                                sessionFieldWithinDay = ['Date_' strrep(sessionTestStr, '-', '_')];
                                undrscr_source = strfind(thisSourceArea, '_test');
                                undrscr_target = strfind(thisTargetArea, '_test');
                                withinDayStruct.(thisMouse).(thisSourceArea(1:undrscr_source-1)).(thisTargetArea(1:undrscr_target-1)).(sessionFieldWithinDay).optDimPerf = y(optDimReducedRankRegress);
                                withinDayStruct.(thisMouse).(thisSourceArea(1:undrscr_source-1)).(thisTargetArea(1:undrscr_target-1)).(sessionFieldWithinDay).optDim = optDimReducedRankRegress;
                                withinDayStruct.(thisMouse).(thisSourceArea(1:undrscr_source-1)).(thisTargetArea(1:undrscr_target-1)).(sessionFieldWithinDay).fullPerf = y_full;
                                withinDayStruct.(thisMouse).(thisSourceArea(1:undrscr_source-1)).(thisTargetArea(1:undrscr_target-1)).(sessionFieldWithinDay).targetDimFA = qOptDimTargetFA;

                            end
                            
                            
                            
                        end
                       
                    end
                end
            end
            
        end
        close all
    end
    
end

%%




%%
% save(fullfile(dataFolder,'sum_struct_drift_80.mat'), 'sumStruct', '-v7.3');
%% Change sumStruct fields to only have RSP (combine RSP and RSPp)

%% Plot summary

sourceAreas = fieldnames(sumStruct);
% vars2plot = {'fracPerf','avgPerf','optDim','avgWithinDayPerf'};
vars2plot = {'fracPerf'};


for iVar = 1:numel(vars2plot)
    
    thisVar = vars2plot{iVar};
    
    for iSource=1:numel(sourceAreas)
        
        thisSource = sourceAreas{iSource};
        targetAreas = fieldnames(sumStruct.(thisSource));
        
        for iTarget=1:numel(targetAreas)
            figure('WindowStyle', 'docked');
            
            thisTarget = targetAreas{iTarget};
            thisPairData = sumStruct.(thisSource).(thisTarget);
            theseDdays = unique(thisPairData.dday);
            thisBPerf = nan(2,size(theseDdays,2));
            thisTheta = nan(2,size(theseDdays,2));
            
            for iDday = 1:numel(theseDdays)
                
                thisDday = theseDdays(iDday);
                sampleSize = nansum(thisPairData.dday==thisDday);
                
                if iVar==3
                    thisDataBPerf = thisPairData.(thisVar)(:,thisPairData.dday==thisDday);
                    thisDataBPerf = thisDataBPerf(:);
                    sampleSize = size(thisDataBPerf,1);
%                     thisDataBPerf = thisDataBPerf(2,:) - thisDataBPerf(1,:);
                else
                    thisDataBPerf = thisPairData.(thisVar)(thisPairData.dday==thisDday);
                end
                
                thisBPerf(1,iDday) = nanmean(thisDataBPerf);
                
                if sampleSize>1
                    
                    thisBPerf(2,iDday) = nanstd(thisDataBPerf) / sqrt(sampleSize);
                    
                end
            end
            
            
            
            %        plot(theseDdays,thisCorr(1,:),'.-','linewidth',2);
            hold on
            errorbar(theseDdays,thisBPerf(1,:),thisBPerf(2,:),'linewidth',2)
            
            xlabel('Delta days')
            ylabel(thisVar)
            if iVar ==1
                ylim([0 1])
                
            elseif iVar==2
                ylim([0 0.075])
                
            elseif iVar==3
                ylim([0 20])
                
            elseif iVar==4
                ylim([0 0.21])    
            end
            xlim([0.5 20.5])
%             
            
            
            box off
            set(gca, 'TickDir', 'out');
            title(['Stabiity of communication subspace for source ' thisSource ' and target ' thisTarget ' using ' thisVar])
            axis square
            
        end
    end
end



%% old code for summary
% sumStruct = struct;
% animalList = fieldnames(dimStruct);
% 
% for iMouse = 1:numel(animalList)
%     
%     thisMouse = animalList{iMouse};
%     sessionList = fieldnames(dimStruct.(thisMouse));
%     
%     for iSessionPair = 1:numel(sessionList)
%         
%         thisSessionPair = sessionList{iSessionPair};
%         
%         thisSessionAreasFull = fieldnames(dimStruct.(thisMouse).(thisSessionPair));
%         thisSessionAreasSource = thisSessionAreasFull(~contains(thisSessionAreasFull,'test'));
%         
%         for iAreaSource = 1:numel(thisSessionAreasSource)
%             
%             thisSource = thisSessionAreasSource{iAreaSource};
%             
%             thisSessionAreasTarget = fieldnames(dimStruct.(thisMouse).(thisSessionPair).(thisSource));
%             
%             for iAreaTarget = 1:numel(thisSessionAreasTarget)
%                 
%                 thisTarget = thisSessionAreasTarget{iAreaTarget};
%                 
%                 dataDay1 = dimStruct.(thisMouse).(thisSessionPair).(thisSource).(thisTarget);
%                 dataDay2 = dimStruct.(thisMouse).(thisSessionPair).([thisSource '_test']).([thisTarget '_test']);
%                 
%                 [loss1, ~] = RegressPredict(dataDay1.Y_target, dataDay1.X_source, dataDay2.B, 'LossMeasure', 'NSE');
%                 [loss2, ~] = RegressPredict(dataDay2.Y_target, dataDay2.X_source, dataDay1.B, 'LossMeasure', 'NSE');
%                 
%                 perf1 = (1-loss1);
%                 perf2 = (1-loss2);
%                 fract_perf1 = perf1/dataDay1.optDimPerf;
%                 fract_perf2 = perf2/dataDay2.optDimPerf;
%                 
%                 
%                 if ~isfield(sumStruct,thisSource)
%                     sumStruct.(thisSource).(thisTarget).fracPerf = [];
%                     sumStruct.(thisSource).(thisTarget).dday = [];
%                 elseif ~isfield(sumStruct.(thisSource),thisTarget)
%                     sumStruct.(thisSource).(thisTarget).fracPerf = [];
%                     sumStruct.(thisSource).(thisTarget).dday = [];
%                 end
%                 
%                 
%                 undrscIdx = strfind(thisSessionPair, '_');
%                 Date1 = thisSessionPair(undrscIdx(1)+1:undrscIdx(2)-1);
%                 Date2 = thisSessionPair(undrscIdx(2)+1:end);
%                 ddays = days(datetime(Date2, 'InputFormat', 'yyyyMMdd')-datetime(Date1, 'InputFormat', 'yyyyMMdd'));
%                 
%                 if dataDay1.optDimPerf>=fit_thresh && dataDay2.optDimPerf>=fit_thresh
%                     
%                     perf1(perf1<0) = 0;
%                     perf2(perf2<0) = 0;
%                     perf1(perf1>1) = 1;
%                     perf2(perf2>1) = 1;
%                     fract_perf1(fract_perf1<0) = 0;
%                     fract_perf2(fract_perf2<0) = 0;
%                     fract_perf1(fract_perf1>1) = 1;
%                     fract_perf2(fract_perf2>1) = 1;
%                     
%                     sumStruct.(thisSource).(thisTarget).fracPerf = [sumStruct.(thisSource).(thisTarget).fracPerf mean([fract_perf1 fract_perf2])];
%                     sumStruct.(thisSource).(thisTarget).dday = [sumStruct.(thisSource).(thisTarget).dday ddays];
%                     sumStruct.(thisSource).(thisTarget).avgPerf = [sumStruct.(thisSource).(thisTarget).avgPerf mean([perf1 perf2])];
%                     
%                 end
%                 
%                 
%             end
%         end
%     end
% end
