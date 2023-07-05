%%
clear all
SET_CONSTS
maxNeuTrgt = 35;
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

%%
for iMouse = 1:numel(animalList)
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
                            
                            % Vector containing the interaction dimensionalities to use when fitting
                            % RRR. 0 predictive dimensions results in using the mean for prediction.
                            numDimsUsedForPrediction = 1:20;
                            
                            % Number of cross validation folds.
                            cvNumFolds = 10;
                            
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
                            
                            cvNumFolds = 10;
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
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).B = B;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).B_ = B_;
                            dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea).V = V;
                        end
                    end
                end
            end
            
        end
        close all
    end
end

%%
save(fullfile(dataFolder,'data_struct_80.mat'), 'dimStruct');

%% Plot single session examples
areaSource = 'PPC';
areaTarget = 'V1';
thisMouse = 'SS33';
sessionField = 'Dates_20191104_20191106' %'Dates_20200224_20200303';
dim1 = dimStruct.(thisMouse).(sessionField).(areaSource).(areaTarget).optDim;
dim2 = dimStruct.(thisMouse).(sessionField).([areaSource '_test']).([areaTarget '_test']).optDim;
B1 = dimStruct.(thisMouse).(sessionField).(areaSource).(areaTarget).B;
B2 = dimStruct.(thisMouse).(sessionField).([areaSource '_test']).([areaTarget '_test']).B;
subspace1 = dimStruct.(thisMouse).(sessionField).(areaSource).(areaTarget).B_(:,1:dim1);
subspace2 = dimStruct.(thisMouse).(sessionField).([areaSource '_test']).([areaTarget '_test']).B_(:,1:dim2);
theta = rad2deg(subspace(subspace1,subspace2))
undrscIdx = strfind(sessionField, '_');
Date1 = sessionField(undrscIdx(1)+1:undrscIdx(2)-1);
Date2 = sessionField(undrscIdx(2)+1:end);


figure;
subplot(2,2,1)
imagesc(B1)
xlabel(areaTarget)
ylabel(areaSource)
title(['Weight matrix on date ' Date1])

subplot(2,2,2)
imagesc(B2)
xlabel(areaTarget)
ylabel(areaSource)
title(['Weight matrix on date ' Date2])


subplot(2,2,3)
hold on
plot([0 1],[0 1],'--k');
plot(B1(:),B2(:),'.');
axis tight
xlabel(['B on day ' Date1])
ylabel(['B on day ' Date2])
title(['Correlation across days is ' num2str(corr(B1(:),B2(:)))])

%% Summarize correlations between B weights
sumStruct = struct;
mouseNames = fieldnames(dimStruct);

for iMouse = 1:numel(mouseNames)
    thisMouse = mouseNames{iMouse};
    sessionNames = fieldnames(dimStruct.(thisMouse));
    
    for iSession = 1:numel(sessionNames)
        thisSession = sessionNames{iSession};
        sourceAreasFull = fieldnames(dimStruct.(thisMouse).(thisSession));
        sourceAreas = sourceAreasFull(~contains(sourceAreasFull,'test'));
        
        for iSource = 1:numel(sourceAreas)
            thisSource = sourceAreas{iSource};
            targetAreasFull = fieldnames(dimStruct.(thisMouse).(thisSession).(thisSource));
            targetAreas = targetAreasFull(~contains(targetAreasFull,'test'));
            
            for iTarget = 1:numel(targetAreas)
                thisTarget = targetAreas{iTarget};
                if ~isfield(sumStruct,thisSource)
                    sumStruct.(thisSource).(thisTarget).corrB = [];
                    sumStruct.(thisSource).(thisTarget).dday = [];
                    sumStruct.(thisSource).(thisTarget).theta = [];
                elseif ~isfield(sumStruct.(thisSource),thisTarget)
                    sumStruct.(thisSource).(thisTarget).corrB = [];
                    sumStruct.(thisSource).(thisTarget).dday = [];
                    sumStruct.(thisSource).(thisTarget).theta = [];
                end
                
                dim1 = dimStruct.(thisMouse).(thisSession).(thisSource).(thisTarget).optDim;
                dim2 = dimStruct.(thisMouse).(thisSession).([thisSource '_test']).([thisTarget '_test']).optDim;
                B1 = dimStruct.(thisMouse).(thisSession).(thisSource).(thisTarget).B;
                B2 = dimStruct.(thisMouse).(thisSession).([thisSource '_test']).([thisTarget '_test']).B;
                subspace1 = dimStruct.(thisMouse).(thisSession).(thisSource).(thisTarget).B_(:,1:dim1);
                subspace2 = dimStruct.(thisMouse).(thisSession).([thisSource '_test']).([thisTarget '_test']).B_(:,1:dim2);
                theta = rad2deg(subspace(subspace1,subspace2));
                undrscIdx = strfind(thisSession, '_');
                Date1 = thisSession(undrscIdx(1)+1:undrscIdx(2)-1);
                Date2 = thisSession(undrscIdx(2)+1:end);
                thisCorrB = corr(B1(:),B2(:));
                ddays = days(datetime(Date2, 'InputFormat', 'yyyyMMdd')-datetime(Date1, 'InputFormat', 'yyyyMMdd'));
                sumStruct.(thisSource).(thisTarget).corrB = [sumStruct.(thisSource).(thisTarget).corrB thisCorrB];
                sumStruct.(thisSource).(thisTarget).dday = [sumStruct.(thisSource).(thisTarget).dday ddays];
                sumStruct.(thisSource).(thisTarget).theta = [sumStruct.(thisSource).(thisTarget).theta theta];
                
            end
        
        end
    end
end

%% Plot summary

sourceAreas = fieldnames(sumStruct);
plotAngle = 0;

for iSource=1:numel(sourceAreas)

   thisSource = sourceAreas{iSource};
   targetAreas = fieldnames(sumStruct.(thisSource));
   
   for iTarget=1:numel(targetAreas)
       figure('WindowStyle', 'docked');
       
       thisTarget = targetAreas{iTarget};
       thisPairData = sumStruct.(thisSource).(thisTarget);
       theseDdays = unique(thisPairData.dday);
       thisCorr = nan(2,size(theseDdays,2));
       thisTheta = nan(2,size(theseDdays,2));
       
       for iDday = 1:numel(theseDdays)
           
           thisDday = theseDdays(iDday);
           sampleSize = nansum(thisPairData.dday==thisDday);
           thisDataCorr = thisPairData.corrB(thisPairData.dday==thisDday);
           thisDataTheta = thisPairData.theta(thisPairData.dday==thisDday);
           thisCorr(1,iDday) = nanmean(thisDataCorr);
           thisTheta(1,iDday) = nanmean(thisDataTheta);
           
           if sampleSize>1
               
               thisCorr(2,iDday) = nanstd(thisDataCorr) / sqrt(sampleSize);
               thisTheta(2,iDday) = nanstd(thisDataTheta) / sqrt(sampleSize);
            
           end
       end
       
       if plotAngle
        yyaxis left
       end
       
%        plot(theseDdays,thisCorr(1,:),'.-','linewidth',2);
       hold on
       errorbar(theseDdays,thisCorr(1,:),thisCorr(2,:),'linewidth',2)
      
       xlabel('Delta days')
       ylabel('mean corr B')
       xlim([0.5 15])
       ylim([0 1])
       
       if plotAngle
           yyaxis right
           plot(theseDdays,thisTheta(1,:),'.-','linewidth',2);
           hold on
           errorbar(theseDdays,thisTheta(1,:),thisTheta(2,:),'linewidth',2)
           ylabel('mean subspace angle')
       end
       box off
       set(gca, 'TickDir', 'out');
       title(['B corr for source ' thisSource ' and target ' thisTarget])
       axis square
              
   end
end
    %%
areasList = {'V1','RSPp','RSP','PPC','M2','S1','M1'};
animalsList = fieldnames(dimStruct);

deltaStruct = struct;

for iAnimal = 1:numel(animalsList)
    thisMouse = animalsList{iAnimal};
    sessionList = fieldnames(dimStruct.(thisMouse));
    
    for iSession = 1:numel(sessionList)
        thisSession = sessionList{iSession};
        
        theseSources = fieldnames(dimStruct.(thisMouse).(thisSession));
        
        for iAreaSource = 1:numel(theseSources)
            thisSourceArea = theseSources{iAreaSource};
            theseTargets = fieldnames(dimStruct.(thisMouse).(thisSession).(thisSourceArea));
            
            for iAreaTarget = 1:numel(theseTargets)
                thisTargetArea = theseTargets{iAreaTarget};
                
                if sum(ismember(theseTargets,thisSourceArea))
                    thisSourceAreaDim = dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisSourceArea).optDim;
                    
                    if strcmp(thisSourceArea,thisTargetArea)==0
                        thisSubspaceDim = dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisTargetArea).optDim;
                        thisSubspacePred = dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisTargetArea).fullPerf;

                        if thisSubspacePred>=0.01
                            thisDelta = thisSourceAreaDim - thisSubspaceDim;
                            if isfield(deltaStruct,thisSourceArea)

                                deltaStruct.(thisSourceArea) = [deltaStruct.(thisSourceArea),thisDelta];
                            else

                                deltaStruct.(thisSourceArea) = thisDelta;

                            end

                        end
                    end
                else
                    
                    
                end
                
                
            end
        end
    end
end


%%

sourceAreasList = fieldnames(deltaStruct);
deltasAll = [];

for iAreaSource=1:numel(sourceAreasList)
    %     figure;
    
    thisSourceArea = sourceAreasList{iAreaSource};
    deltasAll = [deltasAll deltaStruct.(thisSourceArea)];
    %     hist(deltaStruct.(thisSourceArea))
    %     title(thisSourceArea)
end

figure;
histogram(deltasAll)
hold on
plot([0 0],ylim,'--k', 'linewidth',2)
xlim([-4 14]);
xlabel('Delta predictive dimensions')
ylabel('Data sets')
box off
set(gca,'TickDir','out')


%% IGNORE THIS TESTING
%%
matrixTest = randn(10, 20);
theta_same = subspace(matrixTest,matrixTest);
theta_diff = subspace(matrixTest,randn(10, 20)*74);
theta_diff_dim = subspace(matrixTest,randn(10, 50));
figure;
plot([theta_same theta_diff theta_diff_dim])
%%

H = hadamard(8);
A = H(:,2:4);
B = H(:,5:8);
theta = subspace(B,B)
