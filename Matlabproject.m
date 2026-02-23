clearvars -except NaturePaper cfgOverride
clc


cfg = struct();
cfg.seed = 42;
cfg.holdoutFraction = 0.20;


cfg.dataPath = "";
cfg.excelSheet = 1;
cfg.matVarName = "NaturePaper";


cfg.targetCandidates = ["Removal rate (%)", "RemovalRate___", "RemovalRate", "Removal_rate___"];


cfg.idLikeNames = ["Number", "number", "index", "Index", "unnamed: 0", "Unnamed: 0"];


cfg.highThreshold = 90.0;
cfg.targetMaskRange = [];


cfg.enableFeatureDrop = true;
cfg.sparsityThreshold = 0.05;
cfg.nzvThreshold = 0.01;
cfg.corrThreshold = 0.92;
cfg.vifThreshold = 10.0;


cfg.maxCatCardinality = 60;
cfg.catUniqueRatioThreshold = 0.95;
cfg.minCatFrequency = 0.01;


cfg.knnK = 5;


cfg.totalTrials = 80;
cfg.cvFolds = 5;
cfg.minTrialsPerModel = 8;
cfg.dryRun = false;


cfg.enableMoE = true;
cfg.moeClusters = 2;
cfg.moeMinClusterRows = 25;
cfg.moePcaMaxComponents = 6;
cfg.clusterEvalK = 2;


cfg.outDir = "outputs_matlab";
cfg.makePlots = true;


if exist("cfgOverride", "var") && isstruct(cfgOverride)
    cfg = mergeConfig(cfg, cfgOverride);
end


if exist("NaturePaper", "var") && istable(NaturePaper)
    df = NaturePaper;
    inputSource = "workspace:NaturePaper";
elseif strlength(string(cfg.dataPath)) > 0
    df = loadInputTable(cfg.dataPath, cfg.excelSheet, cfg.matVarName);
    inputSource = string(cfg.dataPath);
else
    autoPath = autoPickDataFile();
    if strlength(autoPath) == 0
        error("No input table found. Provide `NaturePaper` in workspace or set cfg.dataPath.");
    end
    df = loadInputTable(autoPath, cfg.excelSheet, cfg.matVarName);
    inputSource = autoPath;
end

if ~istable(df)
    error("Input data must be a table.");
end

if ~ismember("SourceRowID", string(df.Properties.VariableNames))
    df.SourceRowID = (1:height(df))';
end

targetCol = resolveColumnName(df, cfg.targetCandidates);
if strlength(targetCol) == 0
    error("Could not resolve target column. Candidates: %s", strjoin(cfg.targetCandidates, ", "));
end

if ~isempty(cfg.targetMaskRange)
    [df, maskedCount] = applyTargetMask(df, targetCol, cfg.targetMaskRange);
else
    maskedCount = 0;
end

df.(targetCol) = toNumericColumn(df.(targetCol));
validTarget = ~ismissing(df.(targetCol)) & isfinite(df.(targetCol));
df = df(validTarget, :);

predictorVars = setdiff(string(df.Properties.VariableNames), [targetCol, "SourceRowID"], "stable");
dropLeak = false(size(predictorVars));
for i = 1:numel(predictorVars)
    v = predictorVars(i);
    if any(strcmpi(v, cfg.idLikeNames))
        dropLeak(i) = true;
    end
end
leakCols = predictorVars(dropLeak);
predictorVars = predictorVars(~dropLeak);

if isempty(predictorVars)
    error("No predictors available after dropping target and ID-like columns.");
end

yAll = df.(targetCol);
highMaskAll = (yAll >= cfg.highThreshold);
lowMaskAll = ~highMaskAll;
if nnz(highMaskAll) < 2 || nnz(lowMaskAll) < 2
    error("Need at least 2 high and 2 low rows for a valid split (highThreshold=%.2f).", cfg.highThreshold);
end


rng(cfg.seed);
if nnz(highMaskAll) > 10 && nnz(lowMaskAll) > 10
    cv = cvpartition(highMaskAll, "HoldOut", cfg.holdoutFraction);
else
    cv = cvpartition(height(df), "HoldOut", cfg.holdoutFraction);
end

trainMask = training(cv);
testMask = test(cv);

dfTrain = df(trainMask, :);
dfTest = df(testMask, :);

yTrainOrig = dfTrain.(targetCol);
yTestOrig = dfTest.(targetCol);

testHighMask = (yTestOrig >= cfg.highThreshold);
if nnz(testHighMask) < 1
    error("No high-performance rows in test split. Cannot compute high-zone metrics.");
end


[numericVars, categoricalVars, ignoredVars] = splitPredictorTypes(dfTrain, predictorVars);
if ~isempty(ignoredVars)
    error("Unsupported predictor types present: %s", strjoin(ignoredVars, ", "));
end

[dfTrain, dfTest, categoricalVars, catReport] = dropHighCardinalityCategoricals( ...
    dfTrain, dfTest, categoricalVars, cfg.maxCatCardinality, cfg.catUniqueRatioThreshold);

numericVarsStart = numericVars;
if cfg.enableFeatureDrop
    [numericVars, featureReport] = runNumericFeatureSelection(dfTrain, yTrainOrig, numericVars, cfg);
else
    [numericVars, featureReport] = keepNonSparseNumeric(dfTrain, numericVars, cfg.sparsityThreshold);
end

if isempty(numericVars) && isempty(categoricalVars)
    error("All predictors were removed by filtering/feature selection.");
end


[XTrain, XTest, featureNames, prep] = buildDesignMatrices( ...
    dfTrain, dfTest, numericVars, categoricalVars, cfg.minCatFrequency, cfg.knnK);
if isempty(XTrain) || size(XTrain, 2) == 0
    error("Design matrix is empty after preprocessing.");
end

if size(XTrain, 1) ~= numel(yTrainOrig)
    error("XTrain/yTrain length mismatch.");
end
if size(XTest, 1) ~= numel(yTestOrig)
    error("XTest/yTest length mismatch.");
end


yTrainTrans = targetTransform(yTrainOrig);
highThresholdTrans = targetTransform(cfg.highThreshold);

highTrainCount = nnz(yTrainOrig >= cfg.highThreshold);
lowTrainCount = nnz(yTrainOrig < cfg.highThreshold);
XTrainModel = XTrain;
yTrainTransModel = yTrainTrans;

if highTrainCount > 0
    rawRatio = lowTrainCount / highTrainCount;
    weightRatio = min(max(rawRatio * 1.5, 10.0), 1000.0);
else
    weightRatio = 10.0;
end

fprintf("\n===============================================================\n");
fprintf("MATLAB HEATDRAFT PIPELINE\n");
fprintf("===============================================================\n");
fprintf("Input: %s\n", inputSource);
fprintf("Target: %s\n", targetCol);
fprintf("Rows: total=%d, train=%d, test=%d\n", height(df), height(dfTrain), height(dfTest));
fprintf("High threshold: %.2f | train_high=%d train_low=%d test_high=%d\n", ...
    cfg.highThreshold, highTrainCount, lowTrainCount, nnz(testHighMask));
fprintf("Predictors: numeric_start=%d numeric_final=%d categorical_final=%d matrix_cols=%d\n", ...
    numel(numericVarsStart), numel(numericVars), numel(categoricalVars), size(XTrain, 2));
fprintf("Dynamic high-sample weight ratio: %.2f\n", weightRatio);
fprintf("Numeric imputation: KNN (k=%d)\n", cfg.knnK);
fprintf("Training mode: full-train weighted toward high zone\n");

if cfg.dryRun
    fprintf("\nDry run complete. Data/preprocessing checks passed; modeling skipped.\n");
    return;
end


modelDefs = struct( ...
    "Key", {"lsboost", "bag_forest", "neural_net"}, ...
    "DisplayName", {"LSBoost", "BaggedTrees_mtryWide", "NeuralNet"});

perModelTrials = max(cfg.minTrialsPerModel, floor(cfg.totalTrials / numel(modelDefs)));
fprintf("\nTuning base models with weighted CV MAE on transformed target...\n");

baseModels = struct([]);
metricRows = struct([]);
predMap = containers.Map("KeyType", "char", "ValueType", "any");
bestParamsReport = struct();

for i = 1:numel(modelDefs)
    def = modelDefs(i);
    fprintf("- Tuning %s (%d trials)\n", def.DisplayName, perModelTrials);

    [spec, bestCvMae] = tuneModelOptunaLike( ...
        def.Key, XTrainModel, yTrainTransModel, highThresholdTrans, weightRatio, cfg.cvFolds, perModelTrials, cfg.seed + 31 * i);
    spec.DisplayName = string(def.DisplayName);
    spec.CVWeightedMAE = bestCvMae;

    if isempty(baseModels)
        baseModels = spec;
    else
        baseModels(end+1) = spec;
    end

    yHatTestTrans = predict(spec.Model, XTest);
    yHatTest = targetInverseTransform(yHatTestTrans);
    yHatTrainTrans = predict(spec.Model, XTrainModel);
    yHatTrain = targetInverseTransform(yHatTrainTrans);
    predMap(char(spec.DisplayName)) = yHatTest;

    row = evaluateModelRow(spec.DisplayName, yTestOrig, yHatTest, cfg.highThreshold, yTrainOrig, yHatTrain);
    metricRows = addMetricRow(metricRows, row);

    bestParamsReport.(matlab.lang.makeValidName(char(spec.DisplayName))) = spec.Params;
end

baseLeaderboard = sortLeaderboard(struct2table(metricRows));
fprintf("\nBase-model leaderboard (high-zone focus):\n");
disp(baseLeaderboard(:, ["Model", "R2_High", "R2_High_Train", "RMSE_High", "HighHitRate", "R2_Global", "R2_Train"]));


topCount = min(3, height(baseLeaderboard));
topNames = baseLeaderboard.Model(1:topCount);
topSpecs = getSpecsByDisplayName(baseModels, topNames);

stackModel = trainStackingModel(topSpecs, XTrainModel, yTrainTransModel, highThresholdTrans, weightRatio, cfg.cvFolds, cfg.seed + 777);
yHatStackTrans = predictStackingModel(stackModel, XTest);
yHatStack = targetInverseTransform(yHatStackTrans);
yHatStackTrainTrans = predictStackingModel(stackModel, XTrainModel);
yHatStackTrain = targetInverseTransform(yHatStackTrainTrans);

stackRow = evaluateModelRow("stacking_top3", yTestOrig, yHatStack, cfg.highThreshold, yTrainOrig, yHatStackTrain);
metricRows = addMetricRow(metricRows, stackRow);
predMap('stacking_top3') = yHatStack;
bestParamsReport.stacking_top3 = struct("top3", cellstr(topNames(:)));


moeReport = struct("enabled", false, "reason", "Disabled by config");
moeModel = struct();
if cfg.enableMoE
    [moeModel, moeReport] = trainKMeansMoE( ...
        topSpecs(1), XTrainModel, yTrainTransModel, highThresholdTrans, weightRatio, prep.numNumericFeatures, cfg);

    if moeReport.enabled
        yHatMoeTrans = predictKMeansMoE(moeModel, XTest);
        yHatMoe = targetInverseTransform(yHatMoeTrans);
        yHatMoeTrainTrans = predictKMeansMoE(moeModel, XTrainModel);
        yHatMoeTrain = targetInverseTransform(yHatMoeTrainTrans);
        moeRow = evaluateModelRow("moe_kmeans2", yTestOrig, yHatMoe, cfg.highThreshold, yTrainOrig, yHatMoeTrain);
        metricRows = addMetricRow(metricRows, moeRow);
        predMap('moe_kmeans2') = yHatMoe;
    end
end


finalLeaderboard = sortLeaderboard(struct2table(metricRows));
fprintf("\nFinal leaderboard (including stacking and MoE):\n");
disp(finalLeaderboard(:, ["Model", "R2_High", "R2_High_Train", "RMSE_High", "HighHitRate", "R2_Global", "R2_Train"]));

winner = string(finalLeaderboard.Model(1));
winnerKey = char(winner);
if ~isKey(predMap, winnerKey)
    error("Winner '%s' has no predictions registered.", winner);
end
yHatWinner = predMap(winnerKey);


[clusterEval, clusterPerfTable, highClusterPerf] = evaluateWinnerByClusters( ...
    XTrain, yTrainOrig, XTest, yTestOrig, yHatWinner, cfg.clusterEvalK, cfg.seed + 5000);

disp(" ");
disp("===============================================================");
disp("WINNER MODEL: TEST PERFORMANCE BY CLUSTER");
disp("===============================================================");
disp(clusterPerfTable);


[lowDiag, lowGapTable] = buildLowZoneDiagnostics( ...
    yTestOrig, yHatWinner, dfTest, dfTrain, numericVars, targetCol, cfg.highThreshold);


figDir = fullfile(cfg.outDir, "figures");
dataDir = fullfile(cfg.outDir, "data");
reportDir = fullfile(cfg.outDir, "reports");
ensureDir(cfg.outDir);
ensureDir(figDir);
ensureDir(dataDir);
ensureDir(reportDir);

zone = repmat("low", numel(yTestOrig), 1);
zone(yTestOrig >= cfg.highThreshold) = "high";

predTable = table( ...
    dfTest.SourceRowID, yTestOrig(:), yHatWinner(:), ...
    'VariableNames', {'row_id', 'actual', 'predicted'});
predTable.residual = predTable.predicted - predTable.actual;
predTable.abs_error = abs(predTable.residual);
predTable.zone = zone;

metricsPath = fullfile(dataDir, "metrics_leaderboard.csv");
predictionsPath = fullfile(dataDir, "test_predictions.csv");
selectedFeaturesPath = fullfile(dataDir, "selected_features.csv");
calibrationPath = fullfile(dataDir, "calibration_table.csv");
lowGapPath = fullfile(dataDir, "low_zone_feature_gaps.csv");
clusterPerfPath = fullfile(dataDir, "cluster_performance.csv");

writetable(finalLeaderboard, metricsPath);
writetable(predTable, predictionsPath);
selectedFeaturePretty = arrayfun(@(s) prettyFeatureLabel(s), featureNames(:));
writetable(table(featureNames(:), selectedFeaturePretty(:), ...
    'VariableNames', {'feature', 'display_feature'}), selectedFeaturesPath);
writetable(clusterPerfTable, clusterPerfPath);

calibTable = calibrationByQuantile(predTable.actual, predTable.predicted, 8);
writetable(calibTable, calibrationPath);

if isempty(lowGapTable)
    writetable(table(string.empty(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        'VariableNames', {'feature', 'high_train_median', 'low_median', 'abs_gap'}), lowGapPath);
else
    writetable(lowGapTable, lowGapPath);
end

lowDiagPath = fullfile(reportDir, "low_zone_diagnostics.json");
bestParamsPath = fullfile(reportDir, "best_params.json");
splitSummaryPath = fullfile(reportDir, "split_summary.json");
reportPath = fullfile(reportDir, "model_report.json");

splitSummary = struct();
splitSummary.train_rows = height(dfTrain);
splitSummary.test_rows = height(dfTest);
splitSummary.train_high_rows = nnz(yTrainOrig >= cfg.highThreshold);
splitSummary.train_low_rows = nnz(yTrainOrig < cfg.highThreshold);
splitSummary.test_high_rows = nnz(yTestOrig >= cfg.highThreshold);
splitSummary.test_low_rows = nnz(yTestOrig < cfg.highThreshold);
splitSummary.high_threshold = cfg.highThreshold;

writeJsonFile(lowDiagPath, lowDiag);
writeJsonFile(bestParamsPath, bestParamsReport);
writeJsonFile(splitSummaryPath, splitSummary);

artifacts = struct();
artifacts.figures = struct();
artifacts.data = struct();
artifacts.reports = struct();
artifacts.data.metrics_leaderboard = metricsPath;
artifacts.data.test_predictions = predictionsPath;
artifacts.data.selected_features = selectedFeaturesPath;
artifacts.data.calibration_table = calibrationPath;
artifacts.data.low_zone_feature_gaps = lowGapPath;
artifacts.data.cluster_performance = clusterPerfPath;
artifacts.reports.low_zone_diagnostics = lowDiagPath;
artifacts.reports.best_params = bestParamsPath;
artifacts.reports.split_summary = splitSummaryPath;

if cfg.makePlots
    winnerContext = struct();
    winnerContext.baseModels = baseModels;
    winnerContext.stackModel = stackModel;
    winnerContext.moeEnabled = moeReport.enabled;
    winnerContext.moeModel = moeModel;

    [dashboardPath, frontierPath, calibrationFigPath, clusterDetailPath, heatmapPath, featImpPath] = makePlots( ...
        predTable, finalLeaderboard, clusterPerfTable, cfg.highThreshold, figDir, winner, ...
        XTrainModel, yTrainOrig, XTest, yTestOrig, featureNames, winnerContext, cfg.seed);
    artifacts.figures.performance_dashboard = dashboardPath;
    artifacts.figures.model_frontier = frontierPath;
    artifacts.figures.calibration_curve = calibrationFigPath;
    artifacts.figures.cluster_detail = clusterDetailPath;
    artifacts.figures.feature_correlation_heatmap = heatmapPath;
    artifacts.figures.feature_importance_curve = featImpPath;
end

featureSelectionReportOut = struct();
featureSelectionReportOut.enabled = cfg.enableFeatureDrop;
featureSelectionReportOut.n_start_numeric = numel(numericVarsStart);
featureSelectionReportOut.n_final_numeric = numel(numericVars);
featureSelectionReportOut.total_dropped_numeric = numel(numericVarsStart) - numel(numericVars);
featureSelectionReportOut.sparsity_dropped = featureReport.sparsityDropped;
featureSelectionReportOut.nzv_dropped = featureReport.nzvDropped;
featureSelectionReportOut.correlation_dropped = featureReport.corrDropped;
featureSelectionReportOut.vif_dropped = featureReport.vifDropped;

report = struct();
report.input_source = inputSource;
report.target = targetCol;
report.high_threshold = cfg.highThreshold;
report.rows_total = height(df);
report.rows_train = height(dfTrain);
report.rows_test = height(dfTest);
report.high_rows_total = nnz(df.(targetCol) >= cfg.highThreshold);
report.low_rows_total = nnz(df.(targetCol) < cfg.highThreshold);
report.features_raw = numel(predictorVars) + numel(leakCols);
report.features_final_matrix = size(XTrain, 2);
report.winner = winner;
report.masked_target_rows = maskedCount;
report.train_rows_used_for_model = size(XTrainModel, 1);
report.categorical_filter = catReport;
report.feature_selection = featureSelectionReportOut;
report.moe_report = moeReport;
report.metrics = table2struct(finalLeaderboard);
report.cluster_analysis = clusterEval;
report.high_performance_cluster_test = highClusterPerf;
report.low_zone_diagnostics = lowDiag;
report.artifacts = artifacts;

writeJsonFile(reportPath, report);

fprintf("\nSaved report: %s\n", reportPath);
fprintf("Saved predictions: %s\n", predictionsPath);
fprintf("Saved metrics: %s\n", metricsPath);


function tbl = loadInputTable(dataPath, excelSheet, matVarName)
dataPath = string(dataPath);
if ~isfile(dataPath)
    error("Data file not found: %s", dataPath);
end

[~, ~, ext] = fileparts(dataPath);
ext = lower(string(ext));

switch ext
    case ".mat"
        s = load(dataPath);
        if isfield(s, matVarName) && istable(s.(matVarName))
            tbl = s.(matVarName);
            return;
        end
        f = fieldnames(s);
        isTableField = cellfun(@(n) istable(s.(n)), f);
        tableFields = f(isTableField);
        if isscalar(tableFields)
            tbl = s.(tableFields{1});
            return;
        end
        error("No table found in MAT file. Set cfg.matVarName correctly.");

    case {".csv", ".txt"}
        tbl = readtable(dataPath, "VariableNamingRule", "preserve");

    case {".xlsx", ".xls"}
        tbl = readtable(dataPath, "Sheet", excelSheet, "VariableNamingRule", "preserve");

    otherwise
        error("Unsupported file type: %s", ext);
end
end

function picked = autoPickDataFile()
csvFiles = dir("*.csv");
xlsxFiles = dir("*.xlsx");
allFiles = [csvFiles; xlsxFiles];
if isscalar(allFiles)
    picked = string(fullfile(allFiles(1).folder, allFiles(1).name));
else
    picked = "";
end
end

function colName = resolveColumnName(tbl, candidates)
vars = string(tbl.Properties.VariableNames);
candidates = string(candidates);

colName = "";
for i = 1:numel(candidates)
    m = vars(strcmp(vars, candidates(i)));
    if ~isempty(m)
        colName = m(1);
        return;
    end
end

canonVars = strings(size(vars));
for i = 1:numel(vars)
    canonVars(i) = canonicalName(vars(i));
end
for i = 1:numel(candidates)
    c = canonicalName(candidates(i));
    idx = find(canonVars == c, 1, "first");
    if ~isempty(idx)
        colName = vars(idx);
        return;
    end
end
end

function c = canonicalName(s)
s = lower(string(s));
s = regexprep(s, "[^a-z0-9]", "");
c = s;
end

function x = toNumericColumn(v)
if isnumeric(v)
    x = double(v);
    return;
end
if islogical(v)
    x = double(v);
    return;
end
if iscategorical(v)
    v = string(v);
end
if isstring(v)
    x = str2double(v);
    return;
end
if iscell(v)
    x = str2double(string(v));
    return;
end
error("Unsupported target column type: %s", class(v));
end

function [tbl, maskedCount] = applyTargetMask(tbl, targetCol, maskRange)
if numel(maskRange) ~= 2
    error("cfg.targetMaskRange must be empty or [low high].");
end
low = maskRange(1);
high = maskRange(2);
mask = tbl.(targetCol) >= low & tbl.(targetCol) <= high;
maskedCount = nnz(mask);
tbl.(targetCol)(mask) = NaN;
fprintf("Target mask: %d rows masked in [%.4f, %.4f]\n", maskedCount, low, high);
end

function [numVars, catVars, ignoredVars] = splitPredictorTypes(tbl, predictorVars)
numVars = strings(0, 1);
catVars = strings(0, 1);
ignoredVars = strings(0, 1);

for i = 1:numel(predictorVars)
    v = predictorVars(i);
    x = tbl.(v);
    if isnumeric(x) || islogical(x)
        numVars(end+1, 1) = v;
    elseif iscategorical(x) || isstring(x) || ischar(x) || isTextCell(x)
        catVars(end+1, 1) = v;
    else
        ignoredVars(end+1, 1) = v;
    end
end
end

function tf = isTextCell(x)
if ~iscell(x)
    tf = false;
    return;
end
if isempty(x)
    tf = true;
    return;
end
tf = all(cellfun(@(e) ischar(e) || isstring(e) || isempty(e), x));
end

function [trainOut, testOut, catVarsOut, report] = dropHighCardinalityCategoricals( ...
    trainIn, testIn, catVars, maxCardinality, uniqueRatioThreshold)
trainOut = trainIn;
testOut = testIn;

dropped = strings(0, 1);
detailCol = strings(0, 1);
detailNUniq = zeros(0, 1);
detailURatio = zeros(0, 1);
detailIdLike = false(0, 1);

nRows = max(1, height(trainOut));
for i = 1:numel(catVars)
    v = catVars(i);
    s = toStringColumn(trainOut.(v));
    nonMissing = ~ismissing(s);
    uniq = unique(s(nonMissing));
    nUniq = numel(uniq);
    uRatio = nUniq / nRows;
    lowerName = lower(v);
    idLike = contains(lowerName, "id") || contains(lowerName, "name") || ...
        contains(lowerName, "smiles") || contains(lowerName, "inchi") || contains(lowerName, "uuid");

    if nUniq > maxCardinality || uRatio >= uniqueRatioThreshold || (idLike && nUniq > 20)
        dropped(end+1, 1) = v;
        detailCol(end+1, 1) = v;
        detailNUniq(end+1, 1) = nUniq;
        detailURatio(end+1, 1) = uRatio;
        detailIdLike(end+1, 1) = idLike;
    end
end

if ~isempty(dropped)
    trainOut(:, dropped) = [];
    testOut(:, dropped) = [];
end

catVarsOut = setdiff(catVars, dropped, "stable");
report = struct();
report.dropped_columns = cellstr(dropped(:));
if isempty(detailCol)
    report.details = struct("column", {}, "n_unique_train", {}, "unique_ratio_train", {}, "id_like", {});
else
    report.details = table2struct(table(detailCol, detailNUniq, detailURatio, detailIdLike, ...
        'VariableNames', {'column', 'n_unique_train', 'unique_ratio_train', 'id_like'}));
end
end

function [numVarsOut, report] = keepNonSparseNumeric(trainTbl, numVarsIn, sparsityThreshold)
numVarsOut = strings(0, 1);
droppedSparse = strings(0, 1);

for i = 1:numel(numVarsIn)
    v = numVarsIn(i);
    x = safeNumeric(trainTbl.(v));
    ratio = nnz(isfinite(x)) / max(1, numel(x));
    if ratio >= sparsityThreshold
        numVarsOut(end+1, 1) = v;
    else
        droppedSparse(end+1, 1) = v;
    end
end

report = struct();
report.sparsityDropped = cellstr(droppedSparse(:));
report.nzvDropped = {};
report.corrDropped = {};
report.vifDropped = {};
end

function [numVarsOut, report] = runNumericFeatureSelection(trainTbl, yTrain, numVarsIn, cfg)
report = struct();
report.sparsityDropped = {};
report.nzvDropped = {};
report.corrDropped = {};
report.vifDropped = {};

vars = numVarsIn(:);


keepMask = false(size(vars));
for i = 1:numel(vars)
    x = safeNumeric(trainTbl.(vars(i)));
    keepMask(i) = (nnz(isfinite(x)) / max(1, numel(x))) >= cfg.sparsityThreshold;
end
report.sparsityDropped = cellstr(vars(~keepMask));
vars = vars(keepMask);

if isempty(vars)
    numVarsOut = vars;
    return;
end


[X, validVars] = numericMatrixImputed(trainTbl, vars);
vars = validVars;

if isempty(vars)
    numVarsOut = vars;
    return;
end


sd = std(X, 0, 1);
sd(sd == 0) = 1;
Xz = (X - mean(X, 1)) ./ sd;
v = var(Xz, 0, 1);
nzvMask = v < cfg.nzvThreshold;
report.nzvDropped = cellstr(vars(nzvMask));
vars = vars(~nzvMask);
X = X(:, ~nzvMask);

if isempty(vars)
    numVarsOut = vars;
    return;
end


[vars, droppedCorr] = corrPruneFeaturesImputed(X, yTrain, vars, cfg.corrThreshold);
report.corrDropped = cellstr(droppedCorr(:));

if isempty(vars)
    numVarsOut = vars;
    return;
end

[X, validVars] = numericMatrixImputed(trainTbl, vars);
vars = validVars;
if isempty(vars)
    numVarsOut = vars;
    return;
end


[vars, droppedVif] = dropHighVIF(X, vars, cfg.vifThreshold);
report.vifDropped = cellstr(droppedVif(:));

numVarsOut = vars;
end
function [X, varsOut] = numericMatrixImputed(tbl, varsIn)
varsIn = varsIn(:);
X = zeros(height(tbl), numel(varsIn));
keep = false(size(varsIn));

for i = 1:numel(varsIn)
    x = safeNumeric(tbl.(varsIn(i)));
    x(~isfinite(x)) = NaN;
    med = median(x, "omitnan");
    if ~isfinite(med)
        med = 0;
    end
    x = fillmissing(x, "constant", med);

    if std(x, 0, 1) == 0
        keep(i) = false;
    else
        keep(i) = true;
    end
    X(:, i) = x;
end

varsOut = varsIn(keep);
X = X(:, keep);
end

function x = safeNumeric(v)
if isnumeric(v) || islogical(v)
    x = double(v(:));
elseif isstring(v) || iscategorical(v) || iscell(v)
    x = str2double(string(v(:)));
else
    error("Variable type not convertible to numeric: %s", class(v));
end
end

function [XtrOut, XteOut] = imputeNumericTrainTest(XtrIn, XteIn, k)
XtrOut = double(XtrIn);
XteOut = double(XteIn);

if isempty(XtrOut)
    return;
end

XtrOut(~isfinite(XtrOut)) = NaN;
XteOut(~isfinite(XteOut)) = NaN;

trainMed = median(XtrOut, 1, "omitnan");
trainMed(~isfinite(trainMed)) = 0;


XtrBase = XtrOut;
XteBase = XteOut;
for j = 1:size(XtrBase, 2)
    XtrBase(:, j) = fillmissing(XtrBase(:, j), "constant", trainMed(j));
    XteBase(:, j) = fillmissing(XteBase(:, j), "constant", trainMed(j));
end

k = max(1, round(double(k)));
p = size(XtrOut, 2);

for j = 1:p
    yTrain = XtrOut(:, j);
    yTest = XteOut(:, j);

    obsTrain = isfinite(yTrain);
    if nnz(obsTrain) == 0
        continue;
    end

    Xcand = XtrBase(obsTrain, :);
    ycand = yTrain(obsTrain);

    missTrain = find(~isfinite(yTrain));
    for ii = missTrain(:)'
        d = pdist2(XtrBase(ii, :), Xcand);
        good = isfinite(d);
        if ~any(good)
            continue;
        end
        d = d(good);
        vals = ycand(good);
        [~, ord] = sort(d, "ascend");
        take = ord(1:min(k, numel(ord)));
        v = mean(vals(take), "omitnan");
        if isfinite(v)
            XtrOut(ii, j) = v;
        end
    end

    missTest = find(~isfinite(yTest));
    for ii = missTest(:)'
        d = pdist2(XteBase(ii, :), Xcand);
        good = isfinite(d);
        if ~any(good)
            continue;
        end
        d = d(good);
        vals = ycand(good);
        [~, ord] = sort(d, "ascend");
        take = ord(1:min(k, numel(ord)));
        v = mean(vals(take), "omitnan");
        if isfinite(v)
            XteOut(ii, j) = v;
        end
    end
end


for j = 1:size(XtrOut, 2)
    XtrOut(:, j) = fillmissing(XtrOut(:, j), "constant", trainMed(j));
    XteOut(:, j) = fillmissing(XteOut(:, j), "constant", trainMed(j));
end
end

function [selectedVars, droppedVars] = corrPruneFeaturesImputed(X, y, varNames, threshold)
varNames = varNames(:);
R = corr(X, "Rows", "pairwise");
rt = corr(X, y, "Rows", "pairwise");
rt(~isfinite(rt)) = -inf;

n = numel(varNames);
processed = false(1, n);
dropped = false(1, n);

for i = 1:n
    if processed(i)
        continue;
    end
    group = i;
    for j = (i + 1):n
        if processed(j)
            continue;
        end
        if abs(R(i, j)) > threshold
            group(end+1) = j;
        end
    end

    if numel(group) > 1
        [~, bestIdx] = max(abs(rt(group)));
        winner = group(bestIdx);
        losers = setdiff(group, winner);
        dropped(losers) = true;
        processed(group) = true;
    else
        processed(i) = true;
    end
end

selectedVars = varNames(~dropped);
droppedVars = varNames(dropped);
end

function [keptVars, droppedVars] = dropHighVIF(X, varNames, threshold)
varNames = varNames(:);
keepIdx = 1:numel(varNames);
droppedIdx = [];

while numel(keepIdx) >= 2
    Xk = X(:, keepIdx);
    vifs = computeVIFs(Xk);
    [maxVif, worstLocal] = max(vifs);
    if ~isfinite(maxVif) || maxVif > threshold
        droppedIdx(end+1) = keepIdx(worstLocal);
        keepIdx(worstLocal) = [];
    else
        break;
    end
end

keptVars = varNames(keepIdx);
droppedVars = varNames(droppedIdx);
end

function vifs = computeVIFs(X)
[n, p] = size(X);
vifs = nan(1, p);

for j = 1:p
    y = X(:, j);
    others = X(:, setdiff(1:p, j));

    if std(y, 0, 1) == 0
        vifs(j) = inf;
        continue;
    end
    if isempty(others)
        vifs(j) = 1;
        continue;
    end

    Xreg = [ones(n, 1), others];
    b = Xreg \ y;
    yHat = Xreg * b;

    ssRes = sum((y - yHat).^2);
    ssTot = sum((y - mean(y)).^2);
    if ssTot <= eps
        vifs(j) = inf;
        continue;
    end
    r2 = 1 - ssRes / ssTot;
    if r2 >= 0.999999
        vifs(j) = inf;
    else
        vifs(j) = 1 / (1 - r2);
    end
end
end

function [XTrain, XTest, featureNames, prep] = buildDesignMatrices(trainTbl, testTbl, numVars, catVars, minCatFreq, knnK)
nTrain = height(trainTbl);
nTest = height(testTbl);

XTrainParts = {};
XTestParts = {};
nameParts = {};


if ~isempty(numVars)
    XnTrain = zeros(nTrain, numel(numVars));
    XnTest = zeros(nTest, numel(numVars));
    for i = 1:numel(numVars)
        XnTrain(:, i) = safeNumeric(trainTbl.(numVars(i)));
        XnTest(:, i) = safeNumeric(testTbl.(numVars(i)));
    end

    [XnTrain, XnTest] = imputeNumericTrainTest(XnTrain, XnTest, knnK);

    for i = 1:numel(numVars)
        xTr = XnTrain(:, i);
        xTe = XnTest(:, i);
        v = numVars(i);

        med = median(xTr, "omitnan");
        if ~isfinite(med)
            med = 0;
        end
        q75 = prctile(xTr, 75);
        q25 = prctile(xTr, 25);
        iqrVal = q75 - q25;
        if ~isfinite(iqrVal) || iqrVal == 0
            iqrVal = 1;
        end

        xTr = (xTr - med) ./ iqrVal;
        xTe = (xTe - med) ./ iqrVal;

        XTrainParts{end+1} = reshape(xTr, nTrain, 1);
        XTestParts{end+1} = reshape(xTe, nTest, 1);
        nameParts{end+1} = "num__" + v;
    end
end

numFeatureCount = numel(nameParts);


for i = 1:numel(catVars)
    v = catVars(i);
    tr = toStringColumn(trainTbl.(v));
    te = toStringColumn(testTbl.(v));

    tr(ismissing(tr)) = "<missing>";
    te(ismissing(te)) = "<missing>";

    [u, ~, idx] = unique(tr);
    counts = accumarray(idx, 1);
    minCount = max(1, ceil(minCatFreq * nTrain));
    rareCats = u(counts < minCount);
    if ~isempty(rareCats)
        tr(ismember(tr, rareCats)) = "<rare>";
        te(ismember(te, rareCats)) = "<rare>";
    end

    cats = unique(tr, "stable");
    for c = 1:numel(cats)
        catVal = cats(c);
        dTr = double(tr == catVal);
        dTe = double(te == catVal);

        XTrainParts{end+1} = reshape(dTr, nTrain, 1);
        XTestParts{end+1} = reshape(dTe, nTest, 1);

        rawName = "cat__" + v + "__" + catVal;
        nameParts{end+1} = rawName;
    end
end

if isempty(XTrainParts)
    XTrain = [];
    XTest = [];
    featureNames = strings(0, 1);
else
    XTrain = cell2mat(XTrainParts(:)');
    XTest = cell2mat(XTestParts(:)');
    featureNames = string(nameParts(:));
end

prep = struct();
prep.numNumericFeatures = numFeatureCount;
end

function s = toStringColumn(v)
if isstring(v)
    s = v(:);
elseif iscategorical(v)
    s = string(v(:));
elseif ischar(v)
    s = string(cellstr(v));
elseif iscell(v)
    s = string(v(:));
elseif isnumeric(v) || islogical(v)
    s = string(v(:));
else
    error("Unsupported categorical conversion for class: %s", class(v));
end
end

function y = targetTransform(yPct)
epsVal = 1e-4;
yPct = double(yPct(:));

if any(~isfinite(yPct))
    error("Target contains non-finite values before transform.");
end
if any(yPct < 0 | yPct > 100)
    error("Target values must be within [0,100] before transform.");
end

p = min(max(yPct / 100, epsVal), 1 - epsVal);
y = log(p ./ (1 - p));
end

function yPct = targetInverseTransform(yTrans)
yTrans = double(yTrans(:));
yTrans = min(max(yTrans, -30), 30);
p = 1 ./ (1 + exp(-yTrans));
yPct = 100 * p;
end

function [spec, bestScore] = tuneModelOptunaLike(modelType, X, y, highThresholdTrans, weightRatio, cvFolds, nTrials, seed)
rng(seed);

if exist('bayesopt', 'file') ~= 2
    error("bayesopt is required for Optuna-style tuning in MATLAB.");
end

vars = getOptimizableVars(modelType, size(X, 2));
obj = @(T) cvWeightedMae(modelType, tableToParamStruct(T), X, y, highThresholdTrans, weightRatio, cvFolds, seed);

progress = struct();
progress.total = nTrials;

res = bayesopt(obj, vars, ...
    'MaxObjectiveEvaluations', nTrials, ...
    'IsObjectiveDeterministic', true, ...
    'UseParallel', false, ...
    'Verbose', 0, ...
    'PlotFcn', [], ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'OutputFcn', @(results, state) bayesoptProgress(results, state, progress));

bestParams = tableToParamStruct(res.XAtMinObjective);
bestScore = res.MinObjective;

wTrain = ones(size(y));
wTrain(y >= highThresholdTrans) = weightRatio;
mdl = fitModelByType(modelType, bestParams, X, y, wTrain);

spec = struct();
spec.Type = string(modelType);
spec.Params = bestParams;
spec.Model = mdl;
spec.DisplayName = string(modelType);
end

function score = cvWeightedMae(modelType, params, X, y, highThresholdTrans, weightRatio, kFolds, seed)
rng(seed);
n = size(X, 1);
if n < kFolds
    kFolds = max(2, n - 1);
end
cv = cvpartition(n, "KFold", kFolds);

foldMae = nan(cv.NumTestSets, 1);
for f = 1:cv.NumTestSets
    tr = training(cv, f);
    te = test(cv, f);

    Xtr = X(tr, :);
    ytr = y(tr);
    Xte = X(te, :);
    yte = y(te);

    wtr = ones(size(ytr));
    wtr(ytr >= highThresholdTrans) = weightRatio;
    wte = ones(size(yte));
    wte(yte >= highThresholdTrans) = weightRatio;
    mdl = fitModelByType(modelType, params, Xtr, ytr, wtr);
    yHat = predict(mdl, Xte);
    foldMae(f) = weightedMae(yte, yHat, wte);
end

score = mean(foldMae, "omitnan");
if ~isfinite(score)
    score = inf;
end
end

function vars = getOptimizableVars(modelType, numFeatures)
switch lower(string(modelType))
    case "lsboost"
        vars = [
            optimizableVariable('NumLearningCycles', [120, 700], 'Type', 'integer')
            optimizableVariable('LearnRate', [0.01, 0.20], 'Transform', 'log')
            optimizableVariable('MinLeafSize', [2, 20], 'Type', 'integer')
            optimizableVariable('MaxNumSplits', [10, 220], 'Type', 'integer')
        ];
    case "bag_forest"
        lo = max(1, floor(0.3 * numFeatures));
        hi = max(1, numFeatures);
        vars = [
            optimizableVariable('NumLearningCycles', [250, 1400], 'Type', 'integer')
            optimizableVariable('MinLeafSize', [1, 10], 'Type', 'integer')
            optimizableVariable('NumVariablesToSample', [lo, hi], 'Type', 'integer')
        ];
    case "neural_net"
        hi1 = max(16, min(256, 2 * numFeatures));
        hi2 = max(8, min(128, numFeatures));
        vars = [
            optimizableVariable('Layer1', [16, hi1], 'Type', 'integer')
            optimizableVariable('Layer2', [0, hi2], 'Type', 'integer')
            optimizableVariable('Lambda', [1e-7, 1e-1], 'Transform', 'log')
            optimizableVariable('IterationLimit', [120, 900], 'Type', 'integer')
            optimizableVariable('Activation', {'relu', 'tanh'}, 'Type', 'categorical')
        ];
    otherwise
        error("Unknown model type: %s", modelType);
end
end

function params = tableToParamStruct(T)
if istable(T)
    params = table2struct(T);
elseif isstruct(T)
    params = T;
else
    error("Unsupported bayesopt parameter container type: %s", class(T));
end
end

function stop = bayesoptProgress(results, state, progress)
stop = false;
if strcmp(state, 'iteration')
    it = size(results.XTrace, 1);
    best = results.MinObjective;
    pct = 100 * it / max(1, progress.total);
    fprintf("  [%.0f%%] trial %d/%d | best weighted MAE=%.6f\n", pct, it, progress.total, best);
end
end

function mdl = fitModelByType(modelType, params, X, y, w)
if nargin < 5 || isempty(w)
    w = ones(size(y));
end

switch lower(string(modelType))
    case "lsboost"
        t = templateTree("MinLeafSize", params.MinLeafSize, "MaxNumSplits", params.MaxNumSplits);
        mdl = fitrensemble(X, y, "Method", "LSBoost", ...
            "Learners", t, ...
            "NumLearningCycles", params.NumLearningCycles, ...
            "LearnRate", params.LearnRate, ...
            "Weights", w);

    case "bag_forest"
        p = size(X, 2);
        nvs = min(max(1, params.NumVariablesToSample), p);
        t = templateTree("MinLeafSize", params.MinLeafSize, "NumVariablesToSample", nvs);
        mdl = fitrensemble(X, y, "Method", "Bag", ...
            "Learners", t, ...
            "NumLearningCycles", params.NumLearningCycles, ...
            "Weights", w);

    case "neural_net"
        if exist('fitrnet', 'file') ~= 2
            error("fitrnet is required for model type 'neural_net'.");
        end
        l1 = max(1, round(double(params.Layer1)));
        l2 = max(0, round(double(params.Layer2)));
        if l2 > 0
            layerSizes = [l1, l2];
        else
            layerSizes = l1;
        end

        act = string(params.Activation);
        mdl = fitrnet(X, y, ...
            "LayerSizes", layerSizes, ...
            "Activations", act, ...
            "Lambda", double(params.Lambda), ...
            "IterationLimit", round(double(params.IterationLimit)), ...
            "Standardize", true, ...
            "Weights", w);

    otherwise
        error("Unknown model type in fitModelByType: %s", modelType);
end
end

function m = weightedMae(yTrue, yPred, w)
yTrue = yTrue(:);
yPred = yPred(:);
w = w(:);
if numel(yTrue) ~= numel(yPred) || numel(yTrue) ~= numel(w)
    error("weightedMae length mismatch.");
end
m = sum(w .* abs(yTrue - yPred)) / max(eps, sum(w));
end

function rows = addMetricRow(rows, r)
if isempty(rows)
    rows = r;
else
    rows(end+1) = r;
end
end

function row = evaluateModelRow(modelName, yTrue, yPred, highThreshold, yTrainTrue, yTrainPred)
yTrue = yTrue(:);
yPred = yPred(:);
if numel(yTrue) ~= numel(yPred)
    error("evaluateModelRow length mismatch.");
end

globalM = metricCore(yTrue, yPred);
highMask = yTrue >= highThreshold;
highM = metricCore(yTrue(highMask), yPred(highMask));

hasTrain = (nargin >= 6) && ~isempty(yTrainTrue) && ~isempty(yTrainPred);
if hasTrain
    yTrainTrue = yTrainTrue(:);
    yTrainPred = yTrainPred(:);
    if numel(yTrainTrue) ~= numel(yTrainPred)
        error("evaluateModelRow train length mismatch.");
    end
    trainM = metricCore(yTrainTrue, yTrainPred);
    highTrainMask = yTrainTrue >= highThreshold;
    highTrainM = metricCore(yTrainTrue(highTrainMask), yTrainPred(highTrainMask));
else
    trainM = struct("RMSE", NaN, "MAE", NaN, "R2", NaN);
    highTrainM = struct("RMSE", NaN, "MAE", NaN, "R2", NaN);
end

row = struct();
row.Model = string(modelName);
row.RMSE_Global = globalM.RMSE;
row.MAE_Global = globalM.MAE;
row.R2_Global = globalM.R2;
row.RMSE_Train = trainM.RMSE;
row.MAE_Train = trainM.MAE;
row.R2_Train = trainM.R2;
row.RMSE_High = highM.RMSE;
row.MAE_High = highM.MAE;
row.R2_High = highM.R2;
row.RMSE_High_Train = highTrainM.RMSE;
row.MAE_High_Train = highTrainM.MAE;
row.R2_High_Train = highTrainM.R2;
row.HighHitRate = mean(yPred(highMask) >= highThreshold);
row.N_Test = numel(yTrue);
row.N_TestHigh = nnz(highMask);
end

function m = metricCore(yTrue, yPred)
m = struct();
if isempty(yTrue)
    m.RMSE = NaN;
    m.MAE = NaN;
    m.R2 = NaN;
    return;
end
yTrue = yTrue(:);
yPred = yPred(:);
err = yTrue - yPred;
m.RMSE = sqrt(mean(err.^2));
m.MAE = mean(abs(err));
ssRes = sum(err.^2);
ssTot = sum((yTrue - mean(yTrue)).^2);
if ssTot <= 0
    m.R2 = NaN;
else
    m.R2 = 1 - ssRes / ssTot;
end
end

function tbl = sortLeaderboard(tblIn)
tbl = sortrows(tblIn, ["R2_High", "RMSE_High", "HighHitRate"], ["descend", "ascend", "descend"]);
end

function specs = getSpecsByDisplayName(allSpecs, names)
names = string(names(:));
specs = struct([]);

for i = 1:numel(names)
    found = false;
    for j = 1:numel(allSpecs)
        if string(allSpecs(j).DisplayName) == names(i)
            if isempty(specs)
                specs = allSpecs(j);
            else
                specs(end+1) = allSpecs(j);
            end
            found = true;
            break;
        end
    end
    if ~found
        error("Model spec not found for %s", names(i));
    end
end
end

function stack = trainStackingModel(topSpecs, XTrain, yTrain, highThresholdTrans, weightRatio, cvFolds, seed)
rng(seed);
n = size(XTrain, 1);
m = numel(topSpecs);
if m < 1
    error("Need at least one base model for stacking.");
end

oof = nan(n, m);
cv = cvpartition(n, "KFold", min(cvFolds, max(2, n - 1)));

for f = 1:cv.NumTestSets
    tr = training(cv, f);
    te = test(cv, f);

    wTr = ones(nnz(tr), 1);
    yTrFold = yTrain(tr);
    wTr(yTrFold >= highThresholdTrans) = weightRatio;

    for j = 1:m
        spec = topSpecs(j);
        mdl = fitModelByType(spec.Type, spec.Params, XTrain(tr, :), yTrain(tr), wTr);
        oof(te, j) = predict(mdl, XTrain(te, :));
    end
end

if any(~isfinite(oof), "all")
    for j = 1:m
        missing = ~isfinite(oof(:, j));
        if any(missing)
            oof(missing, j) = predict(topSpecs(j).Model, XTrain(missing, :));
        end
    end
end

wMeta = ones(size(yTrain));
wMeta(yTrain >= highThresholdTrans) = weightRatio;
meta = fitrlinear(oof, yTrain, ...
    "Learner", "leastsquares", ...
    "Regularization", "ridge", ...
    "Lambda", 1e-3, ...
    "Weights", wMeta);

stack = struct();
stack.Name = "stacking_top3";
stack.BaseSpecs = topSpecs;
stack.MetaModel = meta;
end

function yHat = predictStackingModel(stack, X)
m = numel(stack.BaseSpecs);
Z = zeros(size(X, 1), m);
for j = 1:m
    Z(:, j) = predict(stack.BaseSpecs(j).Model, X);
end
yHat = predict(stack.MetaModel, Z);
end

function [moe, report] = trainKMeansMoE(baseSpec, XTrain, yTrain, highThresholdTrans, weightRatio, numGateFeatures, cfg)
report = struct("enabled", false, "reason", "Not attempted");
moe = struct();

if numGateFeatures < 2
    report.enabled = false;
    report.reason = "Need at least 2 numeric features for gating.";
    return;
end

gateX = XTrain(:, 1:numGateFeatures);
if size(gateX, 1) < cfg.moeMinClusterRows * cfg.moeClusters
    report.enabled = false;
    report.reason = "Not enough rows for requested MoE cluster sizing.";
    return;
end

nComp = min([cfg.moePcaMaxComponents, size(gateX, 2), size(gateX, 1) - 1]);
if nComp < 1
    report.enabled = false;
    report.reason = "Insufficient rows/features for PCA gating.";
    return;
end

[coeff, score, ~, ~, ~, mu] = pca(gateX);
scoreK = score(:, 1:nComp);

rng(cfg.seed + 9001);
[trainLabels, centroids] = kmeans(scoreK, cfg.moeClusters, "Replicates", 10);

clusterCounts = accumarray(trainLabels, 1, [cfg.moeClusters, 1]);
if any(clusterCounts < cfg.moeMinClusterRows)
    report.enabled = false;
    report.reason = "At least one cluster fell below minimum cluster rows.";
    report.cluster_rows = clusterCounts';
    return;
end

experts = cell(cfg.moeClusters, 1);
for k = 1:cfg.moeClusters
    m = (trainLabels == k);
    w = ones(nnz(m), 1);
    yk = yTrain(m);
    w(yk >= highThresholdTrans) = weightRatio;
    experts{k} = fitModelByType(baseSpec.Type, baseSpec.Params, XTrain(m, :), yk, w);
end

moe.enabled = true;
moe.baseType = baseSpec.Type;
moe.baseParams = baseSpec.Params;
moe.fallbackModel = baseSpec.Model;
moe.experts = experts;
moe.gateIdx = 1:numGateFeatures;
moe.mu = mu;
moe.coeff = coeff(:, 1:nComp);
moe.centroids = centroids;
moe.nComp = nComp;

report.enabled = true;
report.reason = "";
report.n_clusters = cfg.moeClusters;
report.cluster_rows = clusterCounts';
report.gate_numeric_features = numGateFeatures;
end

function yHat = predictKMeansMoE(moe, X)
if ~isfield(moe, "enabled") || ~moe.enabled
    error("MoE model is not enabled.");
end

gateX = X(:, moe.gateIdx);
score = (gateX - moe.mu) * moe.coeff;
labels = assignToCentroids(score, moe.centroids);

yHat = zeros(size(X, 1), 1);
clusters = unique(labels(:))';
for c = clusters
    m = (labels == c);
    if c >= 1 && c <= numel(moe.experts) && ~isempty(moe.experts{c})
        yHat(m) = predict(moe.experts{c}, X(m, :));
    else
        yHat(m) = predict(moe.fallbackModel, X(m, :));
    end
end
end

function labels = assignToCentroids(X, centroids)
D = pdist2(X, centroids);
[~, labels] = min(D, [], 2);
end

function [summary, perfTable, highClusterPerf] = evaluateWinnerByClusters(XTrain, yTrain, XTest, yTest, yHatTest, k, seed)
if nargin < 7
    seed = 42;
end
k = round(double(k));
if k < 2
    error("clusterEvalK must be >= 2.");
end
if size(XTrain, 1) <= k
    error("Not enough TRAIN rows (%d) for k=%d cluster evaluation.", size(XTrain, 1), k);
end

rng(seed);
[trainClusters, centroids] = kmeans(XTrain, k, "Replicates", 10);
testClusters = assignToCentroids(XTest, centroids);

trainClusterMeanTarget = accumarray(trainClusters, yTrain, [k, 1], @mean, NaN);
[highClusterMean, highClusterId] = max(trainClusterMeanTarget);

rows = table();
for c = 1:k
    m = (testClusters == c);
    n = nnz(m);
    if n == 0
        r2 = NaN;
        rmse = NaN;
        mae = NaN;
        meanTarget = NaN;
    else
        met = metricCore(yTest(m), yHatTest(m));
        r2 = met.R2;
        rmse = met.RMSE;
        mae = met.MAE;
        meanTarget = mean(yTest(m));
    end
    r = table(c, n, meanTarget, r2, rmse, mae, c == highClusterId, ...
        'VariableNames', {'Cluster', 'N_Test', 'MeanTarget_Test', 'R2_Test', 'RMSE_Test', 'MAE_Test', 'IsHighPerformanceCluster'});
    rows = [rows; r];
end
perfTable = sortrows(rows, "Cluster");

highMask = (testClusters == highClusterId);
if nnz(highMask) == 0
    error("No TEST rows assigned to high-performance cluster %d; cannot evaluate winner on that cluster.", highClusterId);
end

highMet = metricCore(yTest(highMask), yHatTest(highMask));
highClusterPerf = struct();
highClusterPerf.cluster_id = highClusterId;
highClusterPerf.n_test = nnz(highMask);
highClusterPerf.mean_target_test = mean(yTest(highMask));
highClusterPerf.r2_test = highMet.R2;
highClusterPerf.rmse_test = highMet.RMSE;
highClusterPerf.mae_test = highMet.MAE;
highClusterPerf.train_mean_target = highClusterMean;

summary = struct();
summary.k = k;
summary.high_cluster_id = highClusterId;
summary.train_cluster_mean_target = trainClusterMeanTarget';
summary.test_cluster_counts = accumarray(testClusters, 1, [k, 1], @sum, 0)';
summary.per_cluster_test_metrics = table2struct(perfTable);
end
function calib = calibrationByQuantile(actual, predicted, maxBins)
actual = actual(:);
predicted = predicted(:);
n = numel(actual);

if n < 2
    calib = table(mean(actual), mean(predicted), n, ...
        'VariableNames', {'actual_mean', 'predicted_mean', 'count'});
    return;
end

nBins = min(maxBins, numel(unique(actual)));
if nBins < 2
    calib = table(mean(actual), mean(predicted), n, ...
        'VariableNames', {'actual_mean', 'predicted_mean', 'count'});
    return;
end

q = linspace(0, 1, nBins + 1);
edges = quantile(actual, q);
edges = unique(edges);
if numel(edges) < 3
    calib = table(mean(actual), mean(predicted), n, ...
        'VariableNames', {'actual_mean', 'predicted_mean', 'count'});
    return;
end

bin = discretize(actual, edges, "IncludedEdge", "right");
g = findgroups(bin);

actualMean = splitapply(@mean, actual, g);
predMean = splitapply(@mean, predicted, g);
count = splitapply(@numel, actual, g);
valid = ~isnan(actualMean) & ~isnan(predMean);
calib = table(actualMean(valid), predMean(valid), count(valid), ...
    'VariableNames', {'actual_mean', 'predicted_mean', 'count'});
end

function [lowDiag, gapTable] = buildLowZoneDiagnostics(yTest, yPred, dfTest, dfTrain, numericVars, targetCol, highThreshold)
lowMask = yTest < highThreshold;
if nnz(lowMask) == 0
    lowDiag = struct( ...
        "low_rows", 0, ...
        "false_high_rate", NaN, ...
        "mean_overprediction", NaN, ...
        "low_rmse", NaN, ...
        "low_mae", NaN);
    gapTable = table();
    return;
end

yLowTrue = yTest(lowMask);
yLowPred = yPred(lowMask);
residual = yLowPred - yLowTrue;

lowDiag = struct();
lowDiag.low_rows = nnz(lowMask);
lowDiag.false_high_rate = mean(yLowPred >= highThreshold);
lowDiag.mean_overprediction = mean(residual);
lowDiag.low_rmse = sqrt(mean((yLowTrue - yLowPred).^2));
lowDiag.low_mae = mean(abs(yLowTrue - yLowPred));

if isempty(numericVars)
    gapTable = table();
    return;
end

highTrainMask = dfTrain.(targetCol) >= highThreshold;
feat = strings(0, 1);
highMed = zeros(0, 1);
lowMed = zeros(0, 1);
absGap = zeros(0, 1);
for i = 1:numel(numericVars)
    v = numericVars(i);
    xTrain = safeNumeric(dfTrain.(v));
    xTestLow = safeNumeric(dfTest.(v));

    xTrainHigh = xTrain(highTrainMask);
    xTestLow = xTestLow(lowMask);

    medHigh = median(xTrainHigh, "omitnan");
    medLow = median(xTestLow, "omitnan");

    if ~isfinite(medHigh) || ~isfinite(medLow)
        continue;
    end

    feat(end+1, 1) = string(v);
    highMed(end+1, 1) = medHigh;
    lowMed(end+1, 1) = medLow;
    absGap(end+1, 1) = abs(medLow - medHigh);
end

if isempty(feat)
    gapTable = table();
else
    gapTable = table(feat, highMed, lowMed, absGap, ...
        'VariableNames', {'feature', 'high_train_median', 'low_median', 'abs_gap'});
    gapTable = sortrows(gapTable, "abs_gap", "descend");
    gapTable = gapTable(1:min(12, height(gapTable)), :);
end
end

function out = mergeConfig(baseCfg, overrideCfg)
out = baseCfg;
fn = fieldnames(overrideCfg);
for i = 1:numel(fn)
    out.(fn{i}) = overrideCfg.(fn{i});
end
end

function ensureDir(pathStr)
if ~exist(pathStr, "dir")
    mkdir(pathStr);
end
end

function writeJsonFile(pathStr, s)
txt = jsonencode(s, "PrettyPrint", true);

fid = fopen(pathStr, "w");
if fid < 0
    error("Could not open file for writing: %s", pathStr);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s", txt);
end

function [dashboardPath, frontierPath, calibPath, clusterPath, heatmapPath, featImpPath] = makePlots( ...
    predTable, leaderboard, clusterPerfTable, highThreshold, figDir, winner, ...
    XTrain, yTrain, XTest, yTest, featureNames, winnerContext, seed)
dashboardPath = fullfile(figDir, "performance_dashboard.png");
frontierPath = fullfile(figDir, "model_frontier.png");
calibPath = fullfile(figDir, "calibration_curve.png");
clusterPath = fullfile(figDir, "cluster_detail.png");
heatmapPath = fullfile(figDir, "feature_correlation_heatmap.png");
featImpPath = fullfile(figDir, "feature_importance_curve.png");

colLow = [0.79 0.16 0.16];
colHigh = [0.04 0.45 0.52];
colBlue = [0.11 0.45 0.84];
colWinner = [0.17 0.54 0.24];

actual = predTable.actual(:);
predicted = predTable.predicted(:);
residual = predTable.residual(:);
absErr = predTable.abs_error(:);
isHigh = actual >= highThreshold;


f = figure("Visible", "off", "Color", "w", "Position", [30, 30, 1680, 980]);
tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");

nexttile;
hold on;
scatter(actual(~isHigh), predicted(~isHigh), 20, colLow, "filled", ...
    "MarkerFaceAlpha", 0.50, "MarkerEdgeAlpha", 0.25);
scatter(actual(isHigh), predicted(isHigh), 22, colHigh, "filled", ...
    "MarkerFaceAlpha", 0.60, "MarkerEdgeAlpha", 0.30);
lims = [min([actual; predicted]), max([actual; predicted])];
plot(lims, lims, "k--", "LineWidth", 1.3);
xline(highThreshold, ":", "Color", [0.35 0.35 0.35], "LineWidth", 1.1);
yline(highThreshold, ":", "Color", [0.35 0.35 0.35], "LineWidth", 1.1);
grid on;
xlabel("Actual");
ylabel("Predicted");
title("Predicted vs Actual");
legend("Low zone", "High zone", "Ideal", "Location", "best");
hold off;

nexttile;
hold on;
scatter(actual(~isHigh), residual(~isHigh), 18, colLow, "filled", ...
    "MarkerFaceAlpha", 0.42, "MarkerEdgeAlpha", 0.20);
scatter(actual(isHigh), residual(isHigh), 20, colHigh, "filled", ...
    "MarkerFaceAlpha", 0.48, "MarkerEdgeAlpha", 0.20);
[aSort, ord] = sort(actual);
rSort = residual(ord);
win = max(7, ceil(numel(aSort) / 30));
rTrend = movmean(rSort, win, "omitnan");
plot(aSort, rTrend, "Color", colBlue, "LineWidth", 2.0);
yline(0, "k--", "LineWidth", 1.2);
xline(highThreshold, ":", "Color", [0.35 0.35 0.35], "LineWidth", 1.1);
grid on;
xlabel("Actual");
ylabel("Residual (Predicted - Actual)");
title("Residual Pattern vs Actual");
legend("Low zone", "High zone", "Trend", "Zero", "Location", "best");
hold off;

nexttile;
hold on;
histogram(residual(~isHigh), 26, "Normalization", "probability", ...
    "FaceColor", colLow, "FaceAlpha", 0.56, "EdgeColor", "none");
histogram(residual(isHigh), 26, "Normalization", "probability", ...
    "FaceColor", colHigh, "FaceAlpha", 0.56, "EdgeColor", "none");
xline(0, "k--", "LineWidth", 1.2);
grid on;
xlabel("Residual");
ylabel("Probability");
title("Residual Distribution");
legend("Low zone", "High zone", "Zero", "Location", "best");
hold off;

nexttile;
hit = leaderboard.HighHitRate;
hitScaled = 80 + 220 * (hit - min(hit)) / max(eps, (max(hit) - min(hit)));
hold on;
for i = 1:height(leaderboard)
    if string(leaderboard.Model(i)) == string(winner)
        c = colWinner;
    else
        c = colBlue;
    end
    scatter(leaderboard.RMSE_High(i), leaderboard.R2_High(i), hitScaled(i), c, "filled", ...
        "MarkerFaceAlpha", 0.86, "MarkerEdgeColor", [0.1 0.1 0.1]);
    text(leaderboard.RMSE_High(i), leaderboard.R2_High(i), "  " + string(leaderboard.Model(i)), ...
        "FontSize", 9, "Interpreter", "none");
end
yline(0, "k--", "LineWidth", 1.1);
grid on;
xlabel("RMSE (High)");
ylabel("R2 (High)");
title("Model Frontier (test metrics, size = high hit-rate)");
hold off;

nexttile;
group = categorical(repmat("Low zone", numel(actual), 1), ["Low zone", "High zone"]);
group(isHigh) = "High zone";
boxchart(group, absErr, "BoxFaceColor", [0.35 0.55 0.78], "MarkerStyle", ".");
grid on;
ylabel("Absolute Error");
title("Error Spread by Zone");

nexttile;
wIdx = find(string(leaderboard.Model) == string(winner), 1, "first");
if isempty(wIdx)
    error("Winner '%s' not found in leaderboard for plotting.", winner);
end
trainVals = [leaderboard.R2_Train(wIdx), leaderboard.R2_High_Train(wIdx)];
testVals = [leaderboard.R2_Global(wIdx), leaderboard.R2_High(wIdx)];
bar(categorical(["R2 Global", "R2 High"]), [trainVals(:), testVals(:)], "grouped");
colormap(gca, [0.72 0.79 0.92; 0.12 0.47 0.71]);
yline(0, "k--", "LineWidth", 1.1);
grid on;
ylabel("R2");
title("Winner Train vs Test R2");
legend("Train", "Test", "Location", "best");

sgtitle(sprintf("Performance Dashboard | Winner: %s | R2 test/train = %.3f / %.3f", ...
    winner, leaderboard.R2_Global(wIdx), leaderboard.R2_Train(wIdx)), ...
    "Interpreter", "none", "FontWeight", "bold");
exportgraphics(f, dashboardPath, "Resolution", 190);
close(f);


f2 = figure("Visible", "off", "Color", "w", "Position", [90, 90, 980, 700]);
hold on;
scatter(leaderboard.RMSE_High, leaderboard.R2_High, 110, colBlue, "filled", ...
    "MarkerFaceAlpha", 0.84, "MarkerEdgeColor", [0.1 0.1 0.1]);
[rmseSorted, ord2] = sort(leaderboard.RMSE_High, "ascend");
r2Sorted = leaderboard.R2_High(ord2);
keep = false(size(rmseSorted));
bestR2 = -inf;
for i = 1:numel(rmseSorted)
    if r2Sorted(i) > bestR2
        keep(i) = true;
        bestR2 = r2Sorted(i);
    end
end
plot(rmseSorted(keep), r2Sorted(keep), "-", "Color", [0.55 0.1 0.65], "LineWidth", 2.2);
for i = 1:height(leaderboard)
    txt = sprintf("  %s [test %.3f | train %.3f]", string(leaderboard.Model(i)), ...
        leaderboard.R2_Global(i), leaderboard.R2_Train(i));
    text(leaderboard.RMSE_High(i), leaderboard.R2_High(i), txt, "FontSize", 9, "Interpreter", "none");
end
w = find(string(leaderboard.Model) == string(winner), 1, "first");
if ~isempty(w)
    scatter(leaderboard.RMSE_High(w), leaderboard.R2_High(w), 200, colWinner, "filled", ...
        "MarkerEdgeColor", "k", "LineWidth", 1.0);
end
yline(0, "k--", "LineWidth", 1.1);
grid on;
xlabel("RMSE (High)");
ylabel("R2 (High)");
title("High-Zone Frontier with Pareto Envelope (labels include test/train R2 global)");
legend("Models", "Pareto envelope", "Winner", "Location", "best");
hold off;
exportgraphics(f2, frontierPath, "Resolution", 190);
close(f2);


calibAll = calibrationByQuantile(actual, predicted, 8);
calibLow = calibrationByQuantile(actual(~isHigh), predicted(~isHigh), 6);
calibHigh = calibrationByQuantile(actual(isHigh), predicted(isHigh), 6);

f3 = figure("Visible", "off", "Color", "w", "Position", [110, 110, 1250, 600]);
tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile;
hold on;
scatter(calibAll.actual_mean, calibAll.predicted_mean, 35 + 7 * calibAll.count, colBlue, "filled", ...
    "MarkerEdgeColor", [0.1 0.1 0.1], "MarkerFaceAlpha", 0.8);
plot(calibAll.actual_mean, calibAll.predicted_mean, "-", "Color", colBlue, "LineWidth", 1.8);
lo = min([actual; predicted]);
hi = max([actual; predicted]);
plot([lo, hi], [lo, hi], "k--", "LineWidth", 1.2);
for i = 1:height(calibAll)
    text(calibAll.actual_mean(i), calibAll.predicted_mean(i), sprintf(" n=%d", calibAll.count(i)), "FontSize", 8);
end
grid on;
xlabel("Actual Mean");
ylabel("Predicted Mean");
title("Overall Calibration");
legend("Bin mean", "Curve", "Ideal", "Location", "best");
hold off;

nexttile;
hold on;
plot(calibLow.actual_mean, calibLow.predicted_mean, "-o", "LineWidth", 1.8, ...
    "Color", colLow, "MarkerFaceColor", colLow, "MarkerSize", 6);
plot(calibHigh.actual_mean, calibHigh.predicted_mean, "-o", "LineWidth", 1.8, ...
    "Color", colHigh, "MarkerFaceColor", colHigh, "MarkerSize", 6);
plot([lo, hi], [lo, hi], "k--", "LineWidth", 1.2);
grid on;
xlabel("Actual Mean");
ylabel("Predicted Mean");
title("Zone-Wise Calibration");
legend("Low zone bins", "High zone bins", "Ideal", "Location", "best");
hold off;

sgtitle("Calibration View", "FontWeight", "bold");
exportgraphics(f3, calibPath, "Resolution", 190);
close(f3);


f4 = figure("Visible", "off", "Color", "w", "Position", [130, 130, 1360, 780]);
tiledlayout(2, 2, "TileSpacing", "compact", "Padding", "compact");
clusterIds = clusterPerfTable.Cluster;
highMask = logical(clusterPerfTable.IsHighPerformanceCluster);

nexttile;
bar(clusterIds, clusterPerfTable.R2_Test, "FaceColor", colBlue, "EdgeColor", "k");
hold on;
if any(highMask)
    scatter(clusterIds(highMask), clusterPerfTable.R2_Test(highMask), 140, colHigh, "filled", "MarkerEdgeColor", "k");
end
yline(0, "k--", "LineWidth", 1.1);
grid on;
xlabel("Cluster ID");
ylabel("R2 (Test)");
title("R2 by Cluster");
hold off;

nexttile;
Y = [clusterPerfTable.RMSE_Test, clusterPerfTable.MAE_Test];
bar(clusterIds, Y, "grouped");
grid on;
xlabel("Cluster ID");
ylabel("Error");
title("RMSE and MAE by Cluster");
legend("RMSE", "MAE", "Location", "best");

nexttile;
yyaxis left;
bar(clusterIds, clusterPerfTable.N_Test, 0.8, "FaceColor", [0.77 0.82 0.89], "EdgeColor", "k");
ylabel("Test Rows");
yyaxis right;
plot(clusterIds, clusterPerfTable.MeanTarget_Test, "-o", "Color", colHigh, ...
    "LineWidth", 2.0, "MarkerFaceColor", colHigh);
ylabel("Mean Target (Test)");
grid on;
xlabel("Cluster ID");
title("Cluster Size and Mean Target");

nexttile;
scatter(clusterPerfTable.RMSE_Test, clusterPerfTable.R2_Test, 100, colBlue, "filled");
hold on;
for i = 1:height(clusterPerfTable)
    text(clusterPerfTable.RMSE_Test(i), clusterPerfTable.R2_Test(i), ...
        sprintf("  C%d", clusterPerfTable.Cluster(i)), "FontSize", 9);
end
if any(highMask)
    scatter(clusterPerfTable.RMSE_Test(highMask), clusterPerfTable.R2_Test(highMask), ...
        170, colHigh, "filled", "MarkerEdgeColor", "k");
end
yline(0, "k--", "LineWidth", 1.1);
grid on;
xlabel("RMSE (Test)");
ylabel("R2 (Test)");
title("Cluster Error-R2 Map");
hold off;

sgtitle(sprintf("Cluster Detail View | Winner: %s", winner), "Interpreter", "none", "FontWeight", "bold");
exportgraphics(f4, clusterPath, "Resolution", 190);
close(f4);


nFeat = size(XTrain, 2);
if nFeat < 2
    f5 = figure("Visible", "off", "Color", "w", "Position", [150, 150, 900, 500]);
    text(0.1, 0.5, "Not enough features for correlation heatmap.", "FontSize", 12);
    axis off;
    exportgraphics(f5, heatmapPath, "Resolution", 190);
    close(f5);
else
    rhoY = corr(XTrain, yTrain, "Rows", "pairwise");
    rhoY(~isfinite(rhoY)) = 0;
    [~, ordFeat] = sort(abs(rhoY), "descend");
    topN = min(20, nFeat);
    topIdx = ordFeat(1:topN);
    Xh = XTrain(:, topIdx);
    C = corr(Xh, "Rows", "pairwise");
    C(~isfinite(C)) = 0;

    nameTop = string(featureNames(topIdx));
    nameTop = arrayfun(@(s) truncateLabel(prettyFeatureLabel(s), 30), nameTop);

    f5 = figure("Visible", "off", "Color", "w", "Position", [150, 150, 1350, 680]);
    tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

    nexttile;
    imagesc(C, [-1 1]);
    axis square;
    colormap(gca, parula(256));
    cb = colorbar;
    cb.Label.String = "Pearson correlation";
    xticks(1:topN);
    yticks(1:topN);
    xticklabels(nameTop);
    yticklabels(nameTop);
    set(gca, "TickLabelInterpreter", "none");
    xtickangle(45);
    title(sprintf("Top %d Feature Correlation Heatmap", topN));

    nexttile;
    b = barh(abs(rhoY(topIdx)), "FaceColor", [0.11 0.45 0.84], "EdgeColor", "none");
    b.FaceAlpha = 0.85;
    set(gca, "YDir", "reverse");
    yticks(1:topN);
    yticklabels(nameTop);
    set(gca, "TickLabelInterpreter", "none");
    grid on;
    xlabel("|corr(feature, target)| on train");
    title("Top Features by Target Correlation");

    sgtitle(sprintf("Feature Correlation View | Winner: %s", winner), "Interpreter", "none", "FontWeight", "bold");
    exportgraphics(f5, heatmapPath, "Resolution", 190);
    close(f5);
end


impTbl = computePermutationImportance(winner, winnerContext, XTest, yTest, featureNames, 3, seed + 7000);
topImp = impTbl(1:min(25, height(impTbl)), :);

f6 = figure("Visible", "off", "Color", "w", "Position", [180, 180, 1280, 700]);
tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile;
bar(topImp.PermutationImportance, "FaceColor", [0.04 0.45 0.52], "EdgeColor", "none");
grid on;
xticks(1:height(topImp));
topImpLabels = arrayfun(@(s) truncateLabel(prettyFeatureLabel(s), 30), string(topImp.feature));
xticklabels(topImpLabels);
set(gca, "TickLabelInterpreter", "none");
xtickangle(45);
ylabel("RMSE increase when permuted");
title("Feature Importance Curve (descending)");

nexttile;
topImpLabelsRev = topImpLabels(end:-1:1);
barh(categorical(topImpLabelsRev, topImpLabelsRev), topImp.RelativeImportancePct(end:-1:1), ...
    "FaceColor", [0.79 0.16 0.16], "EdgeColor", "none");
set(gca, "TickLabelInterpreter", "none");
grid on;
xlabel("Relative importance (%)");
title("Relative Importance Share");

sgtitle(sprintf("Winner Feature Importance | %s", winner), "Interpreter", "none", "FontWeight", "bold");
exportgraphics(f6, featImpPath, "Resolution", 190);
close(f6);
end

function impTbl = computePermutationImportance(winnerName, winnerContext, XTest, yTest, featureNames, nRepeats, seed)
rng(seed);
basePred = predictByWinnerName(winnerName, winnerContext, XTest);
baseRmse = sqrt(mean((yTest(:) - basePred(:)).^2));

p = size(XTest, 2);
imp = zeros(p, 1);
for j = 1:p
    delta = zeros(nRepeats, 1);
    for r = 1:nRepeats
        Xp = XTest;
        permIdx = randperm(size(Xp, 1));
        Xp(:, j) = Xp(permIdx, j);
        yp = predictByWinnerName(winnerName, winnerContext, Xp);
        rmseP = sqrt(mean((yTest(:) - yp(:)).^2));
        delta(r) = rmseP - baseRmse;
    end
    imp(j) = max(0, mean(delta, "omitnan"));
end

totalImp = sum(imp);
if totalImp <= 0
    rel = zeros(size(imp));
else
    rel = 100 * imp / totalImp;
end

impTbl = table(string(featureNames(:)), imp, rel, ...
    'VariableNames', {'feature', 'PermutationImportance', 'RelativeImportancePct'});
impTbl = sortrows(impTbl, "PermutationImportance", "descend");
end

function yHatPct = predictByWinnerName(winnerName, winnerContext, X)
winnerName = string(winnerName);
if winnerName == "stacking_top3"
    yHatPct = targetInverseTransform(predictStackingModel(winnerContext.stackModel, X));
    return;
end
if winnerName == "moe_kmeans2"
    if ~isfield(winnerContext, "moeEnabled") || ~winnerContext.moeEnabled
        error("Winner is moe_kmeans2 but MoE context is not enabled.");
    end
    yHatPct = targetInverseTransform(predictKMeansMoE(winnerContext.moeModel, X));
    return;
end

spec = [];
for i = 1:numel(winnerContext.baseModels)
    if string(winnerContext.baseModels(i).DisplayName) == winnerName
        spec = winnerContext.baseModels(i);
        break;
    end
end
if isempty(spec)
    error("Could not resolve winner model '%s' for importance computation.", winnerName);
end
yHatPct = targetInverseTransform(predict(spec.Model, X));
end

function out = truncateLabel(in, maxLen)
s = string(in);
if strlength(s) <= maxLen
    out = s;
else
    out = extractBetween(s, 1, maxLen - 3) + "...";
end
end

function out = prettyFeatureLabel(in)
s = string(in);
if startsWith(s, "num__")
    out = extractAfter(s, "num__");
    return;
end
if startsWith(s, "cat__")
    rest = extractAfter(s, "cat__");
    parts = split(rest, "__");
    if numel(parts) >= 2
        left = parts(1);
        right = strjoin(parts(2:end), " / ");
        out = left + " = " + right;
    else
        out = rest;
    end
    return;
end
out = s;
end
