%%
clear all
SET_CONSTS

% load('mat_sample/sample_data.mat')

% load('mat_sample/test2.mat')
% load('mat_sample/V1.mat')
% load('mat_sample/PPC.mat')
% load('mat_sample/RSPp.mat')
% load('mat_sample/M2.mat')
load('C:\Users\sofia\Documents\GitHub\communication-subspace-matlab\mat_sample\source_id.mat')
load('C:\Users\sofia\Documents\GitHub\communication-subspace-matlab\mat_sample\source_slice.mat')
load('C:\Users\sofia\Documents\GitHub\communication-subspace-matlab\mat_sample\sc_residuals.mat')
load('C:\Users\sofia\Documents\GitHub\communication-subspace-matlab\mat_sample\source_area_label.mat')

% %%
% area_source = 'RSPp';
% area_target = 'RSPp';
% 
% this_source_idx = find(strcmp(area_source,source_area_label));
% data2use = sc_residuals(this_source_idx,:);
%%

data2use = RSPp;

% area_source = 'PPC';
% area_target = 'PPC';
% 
% this_source_idx = find(strcmp(area_source,source_area_label));
% data2use = sc_residuals(this_source_idx,:);

maxNeuTrgt = 50;
maxNeuSrc = 100;

useParallel = 0;
randI = randperm(size(data2use,1));
selectedI = sort(randI(1:maxNeuTrgt));
keepIndices = true(size(data2use,1), 1);
keepIndices(selectedI) = false;
iKeep = find(keepIndices==1);
iKeepRand = randperm(size(iKeep,1));
iKeepRand = iKeep(sort(iKeepRand(1:maxNeuSrc)));
X = data2use(iKeepRand,:)';
Y_V2 = data2use(selectedI,:)';

%% 
data2use = M2;
randI = randperm(size(data2use,1));
selectedI = sort(randI(1:maxNeuTrgt));
Y_V2 = data2use(selectedI,:)';
%%
% %
% X = test1';
% Y_V2 = test2';

% clear test1 test2
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
cvl = crossval(cvFun, Y_V2, X, ...
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

errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
    'MarkerFaceColor', COLOR(V2,:), 'MarkerSize', 10)

xlabel('Number of predictive dimensions')
ylabel('Predictive performance')
%
% Cross-validated Ridge Regression

cvNumFolds = 10;
regressMethod = @RidgeRegress;

% For Ridge regression, the correct range for lambda can be determined
% using:

dMaxShrink = .5:.01:1;
lambda = GetRidgeLambda(dMaxShrink, X);

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

cvl = crossval(cvFun, Y_V2, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

cvLoss = ...
	[ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

[optLambda,optLoss] = ModelSelect...
	(cvLoss, lambda);
%
% Plot Ridge Regression cross-validation results
x = 0;
y = 1-optLoss(1,:);

hold on
plot(x, y, 'o--', 'Color', COLOR(V2,:), ...
    'MarkerFaceColor', 'w', 'MarkerSize', 10)
hold off

xlim([-0.6 numDimsUsedForPrediction(end)+0.6])

%% Cross-validate Factor Regression

numDimsUsedForPrediction = 1:10;

cvNumFolds = 10;

cvOptions = statset('crossval');
if useParallel
    
    cvOptions.UseParallel = true;
end

regressMethod = @FactorRegress;

% In order to apply Factor Regression, we must first determine the optimal
% dimensionality for the Factor Analysis Model
p = size(X, 2);
q = 0:30;
qOpt = FactorAnalysisModelSelect( ...
	CrossValFa(X, q, cvNumFolds, cvOptions), ...
	q);

cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
	numDimsUsedForPrediction, ...
	'LossMeasure', 'NSE', 'qOpt', qOpt);
% qOpt is an extra argument for FactorRegress. Extra arguments for the
% regression function are passed as name/value pairs after the
% cross-validation parameter (in this case numDimsUsedForPrediction).
% qOpt, the optimal factor analysis dimensionality for the source activity
% X, must be provided when cross-validating Factor Regression. When
% absent, Factor Regression will automatically determine qOpt via 
% cross-validation (which will generate an error if Factor Regression is
% itself used within a cross-validation procedure).

cvl = crossval(cvFun, Y_V2, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

cvLoss = ...
	[ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

optDimFactorRegress = ModelSelect...
	(cvLoss, numDimsUsedForPrediction);

% Plot Reduced Rank Regression cross-validation results
x = numDimsUsedForPrediction;
x(x > qOpt) = [];
y = 1-cvLoss(1,:);
e = cvLoss(2,:);

hold on
errorbar(x, y, e, 'o--', 'Color', COLOR(V2,:), ...
    'MarkerFaceColor', 'w', 'MarkerSize', 10)
hold off

legend('Reduced Rank Regression', ...
    'Factor Regression', ...
	'Location', 'SouthEast')




