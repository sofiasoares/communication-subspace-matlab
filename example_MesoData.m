%%
clear all
SET_CONSTS
maxNeuTrgt = 50;
maxNeuSrc = 100;
dim_fract = 0.8;%fraction of full performance to use in estimating dimensionality of subspace
useParallel = 0;
dimStruct = struct;
dataFolder = 'mat_sample';
animalList = dir(dataFolder);
% Filter out non-folders
animalList = animalList([animalList.isdir]);  % Keep only items that are directories
animalList = {animalList.name};  % Store the names in a cell array
animalList = animalList(~ismember(animalList, {'.', '..'}));
animalList = animalList(contains(animalList, 'SS'));


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
        
        % Loop through each file
        for i = 1:numel(files)
            filePath = fullfile(thisSessionPath, files(i).name);
            tmp = load(filePath);
            varName = fieldnames(tmp);
            varName = varName{1};
            thisSessionData.(varName) = tmp.(varName);
        end
        
        this_session_areas = fieldnames(thisSessionData);
        
        for iAreaSource = 1:numel(this_session_areas)
            
            thisSourceArea = this_session_areas{iAreaSource};
            disp(['Working on source area ' thisSourceArea]);
            
            data2use = thisSessionData.(thisSourceArea);
            if size(data2use,1)>=[maxNeuTrgt+maxNeuSrc]
                randI = randperm(size(data2use,1));
                selectedI = sort(randI(1:maxNeuTrgt));
                keepIndices = true(size(data2use,1), 1);
                keepIndices(selectedI) = false;
                iKeep = find(keepIndices==1);
                iKeepRand = randperm(size(iKeep,1));
                iKeepRand = iKeep(sort(iKeepRand(1:maxNeuSrc)));
                X_source = data2use(iKeepRand,:)';
                Y_target_same = data2use(selectedI,:)';
                
                for iAreaTarget = 1:numel(this_session_areas)
                    
                    thisTargetArea = this_session_areas{iAreaTarget};
                    
                    data2use = thisSessionData.(thisTargetArea);
                    
                    if size(data2use,1)>=maxNeuTrgt
                        
                        if strcmp(thisTargetArea,thisSourceArea)==0
                            randI = randperm(size(data2use,1));
                            selectedI = sort(randI(1:maxNeuTrgt));
                            Y_target = data2use(selectedI,:)';
                            
                        elseif strcmp(thisTargetArea,thisSourceArea)==1
                            
                            Y_target = Y_target_same;
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
                        
                        if isempty(this_dim)
                            title([thisSourceArea ' - ' thisTargetArea ' and we could not determine dimensionality'])
                        else
                            
                            title([thisSourceArea ' - ' thisTargetArea ' has ' num2str(this_dim) ' dimensions'])
                        end
                        
                        saveas(gcf,fullfile(pathNameFigs, [thisSourceArea '_' thisTargetArea '_cs.pdf']))
                        sessionField = ['Date' strrep(sessionLabel, '-', '')];
                        dimStruct.(thisMouse).(sessionField).(thisSourceArea).(thisTargetArea)=[this_dim,y_full];
                        
                    end
                end
            end
        end
        close all
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
                    thisSourceAreaDim = dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisSourceArea)(1);
                    
                    if strcmp(thisSourceArea,thisTargetArea)==0
                        if size(dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisTargetArea),2)==2
                            thisSubspaceDim = dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisTargetArea)(1);
                            thisSubspacePred = dimStruct.(thisMouse).(thisSession).(thisSourceArea).(thisTargetArea)(2);

                        if thisSubspacePred>=0.01
                            thisDelta = thisSourceAreaDim- thisSubspaceDim;
                            if isfield(deltaStruct,thisSourceArea)
                                
                                deltaStruct.(thisSourceArea) = [deltaStruct.(thisSourceArea),thisDelta];
                            else
                                
                                deltaStruct.(thisSourceArea) = thisDelta;
                                
                            end
                            
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
hist(deltasAll)
hold on
plot([0 0],ylim,'--r')
xlim([-10 10]);
xlabel('delta predictive dimensions')

