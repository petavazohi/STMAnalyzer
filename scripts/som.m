
clear
%% Settings for Self-Organizing Map
settings = readstruct('settings.json'); 

%% Directory Creation for Saving Figures
todayDate = datestr(now, 'yyyymmdd');
% Format the directory name with various parameters
dirName = sprintf('%s-%s-%dx%d-%s-%dEp-%s-covStp%d-iniNei%d', todayDate, settings.system, settings.dimensions(1), settings.dimensions(2), settings.topologyFcn, settings.epochs, settings.distanceFcn, settings.coverSteps, settings.initNeighbor);

if ~exist(dirName, 'dir')
    mkdir(dirName);
end
%


% Get information about the HDF5 file
info = h5info(settings.filePath);

% Initialize the structure to hold data and attributes
dataStruct = struct();
attributeStruct = struct();

% Loop through each dataset in the file and read data and attributes
for i = 1:length(info.Datasets)
    datasetName = info.Datasets(i).Name; % Get the dataset name
    validDatasetName = matlab.lang.makeValidName(datasetName); % Convert to valid MATLAB field name
    fullPath = strcat('/', datasetName); % Construct the full path for h5read
    
    % Read the dataset into the structure
    dataStruct.(validDatasetName) = h5read(settings.filePath, fullPath);
    
    % Read the attributes for this dataset
    attr_info = info.Datasets(i).Attributes;
    for j = 1:length(attr_info)
        attrName = attr_info(j).Name; % Get the attribute name
        validAttrName = matlab.lang.makeValidName(attrName); % Convert to valid MATLAB field name
        attributeStruct.(validDatasetName).(validAttrName) = h5readatt(settings.filePath, fullPath, attrName); % Read the attribute
    end
end
% clear attr_info attrName datasetName fullPath info i j validAttrName validDatasetName


%% vStack data
fieldNames = fieldnames(dataStruct);
numDatapoints = 0;

for i = 1:numel(fieldNames)
    currentData = dataStruct.(fieldNames{i});
    [zDim, rowCount] = size(currentData);
    numDatapoints = numDatapoints + rowCount;
end
if (settings.dimensions(1) == 0)
    n = round(sqrt(5*sqrt(numDatapoints)));
    settings.dimensions = [n, n];
end

data = zeros(zDim, numDatapoints);
rowIndex = 1;
for i = 1:numel(fieldNames)
    currentData = dataStruct.(fieldNames{i});
    [~, rowCount] = size(currentData);
    data(:, rowIndex:rowIndex + rowCount - 1) = currentData;
    rowIndex = rowIndex + rowCount;
end

% clearvars -except data selfOrgMap settings dataStruct attributeStruct todayDate dirName 
%% Define the Self Organizing Map
selfOrgMap = selforgmap(settings.dimensions, ...
                        'coverSteps', settings.coverSteps, ...
                        'initNeighbor', settings.initNeighbor, ...
                        'topologyFcn', settings.topologyFcn, ...
                        'distanceFcn', settings.distanceFcn);
selfOrgMap.trainParam.epochs = settings.epochs;
% selfOrgMap.trainParam.showWindow = false;
%% Train the SOM
[selfOrgMap,tr] = train(selfOrgMap, data);


figure;
plotsomnd(selfOrgMap);
% Create the 'hot' colormap
hotMap = hot;  % This creates a standard 'hot' colormap

% Reverse the colormap
reversedHotMap = flipud(hotMap);  % This flips the colormap upside down

% Apply the reversed colormap to the current figure
colormap(reversedHotMap);
colorbar;
set(gca, 'XTick', [], 'YTick', []);
% Loop through each file format and save the figure
fileFormats = {'.svg', '.png'};
for i = 1:length(fileFormats)
    % Create the file name with the desired format
    fileName = fullfile(dirName, ['Weight Distances' fileFormats{i}]);

    % Save the figure in the specified format
    if strcmp(fileFormats{i}, '.fig')
        % For .fig, use savefig
        savefig(fileName);
    else
        % For other formats, use print
        print(gcf, fileName, ['-d' erase(fileFormats{i}, '.')], '-r300');
    end
end

scalingFactor = 30; % Adjust this value based on how spacious you want the plot
% Set the figure size
figure('Position', [100, 100, settings.dimensions(1)*scalingFactor, settings.dimensions(2)*scalingFactor]); % [left, bottom, width, height]
plotsomhits(selfOrgMap,data)
set(gca, 'XTick', [], 'YTick', []);
% Loop through each file format and save the figure

for i = 1:length(fileFormats)
    % Create the file name with the desired format
    fileName = fullfile(dirName, ['Hit Histogram' fileFormats{i}]);

    % Save the figure in the specified format
    if strcmp(fileFormats{i}, '.fig')
        % For .fig, use savefig
        savefig(fileName);
    else
        % For other formats, use print
        print(gcf, fileName, ['-d' erase(fileFormats{i}, '.')], '-r300');
    end
end

%% Extract the weights
ws = selfOrgMap.IW{1, 1};
fileName = fullfile(dirName, 'output.hdf5');
h5create(fileName, "/weights", size(ws));
h5write(fileName, '/weights', ws)

%% apply SOM to each dataset

fieldNames = fieldnames(dataStruct);
rowIndex = 1;
for i = 1:numel(fieldNames)
    x = dataStruct.(fieldNames{i});
    y = selfOrgMap(x);
    clusterIndex = vec2ind(y);
    histmp = histcounts(clusterIndex, length(clusterIndex), 'BinMethod','integers')';
    h5create(fileName, "/"+fieldNames{i}+"/hitHistorgam" , size(histmp));
    h5write(fileName, "/"+fieldNames{i}+"/hitHistorgam", histmp);
    
    h5create(fileName, "/"+fieldNames{i}+"/clusterIndex", size(clusterIndex));
    h5write(fileName, "/"+fieldNames{i}+"/clusterIndex", clusterIndex);
end

x = data;
y = selfOrgMap(x);
clusterIndex = vec2ind(y);
histmp = histcounts(clusterIndex, length(clusterIndex), 'BinMethod','integers')';
h5create(fileName, "/all/hitHistogram" , size(histmp));
h5write(fileName, "/all/hitHistogram", histmp);

h5create(fileName, "/all/clusterIndex", size(clusterIndex));
h5write(fileName, "/all/clusterIndex", clusterIndex);
