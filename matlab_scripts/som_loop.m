
clear
%% Settings for Self-Organizing Map
settings = readstruct('settings.json'); 
% TODO, this setting must be stored in the output h5 file. 
%% Directory Creation for Saving Figures
todayDate = datestr(now, 'yyyymmdd');
% Format the directory name with various parameters
baseDir = sprintf('%s-%s-%dx%d-%s-%dEp-%s-covStp%d-iniNei%d', todayDate, settings.system, settings.dimensions(1), settings.dimensions(2), settings.topologyFcn, settings.epochs, settings.distanceFcn, settings.coverSteps, settings.initNeighbor);

if ~exist(baseDir, 'dir')
    mkdir(baseDir);
elseif isfield(settings, 'overWrite') && settings.overWrite
    rmdir(baseDir, 's');
    mkdir(baseDir);
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
if (settings.type=="ind")
    %% vStack data
    fieldNames = fieldnames(dataStruct);
    numDatapoints = 0;

    for iName = 1:numel(fieldNames)
        data = dataStruct.(fieldNames{iName});
        
        dirName = fullfile(baseDir, info.Datasets(iName).Name);
        if ~exist(dirName, 'dir')
            mkdir(dirName);
        end
        %
        [zDim, numDatapoints] = size(data);
        if (settings.dimensions(1) == 0)
            n = round(sqrt(5*sqrt(numDatapoints)));
            settings.dimensions = [n, n];
        end
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

        figure1 = figure;
        plotsomnd(selfOrgMap);
        % % Create the 'hot' colormap
        % hotMap = hot;  % This creates a standard 'hot' colormap
        % % Reverse the colormap
        % reversedHotMap = flipud(hotMap);  % This flips the colormap upside down
        % % Apply the reversed colormap to the current figure
        % colormap(reversedHotMap);
        % colorbar;
        set(gca, 'XTick', [], 'YTick', []);
        % Loop through each file format and save the figure
        for iFormat = 1:length(settings.storeFormats)
            % Create the file name with the desired format
            fileName = fullfile(dirName, ['Weight Distances' settings.storeFormats{iFormat}]);

            % Save the figure in the specified format
            if strcmp(settings.storeFormats{iFormat}, '.fig')
                % For .fig, use savefig
                savefig(fileName);
            else
                % For other formats, use print
                exportgraphics(figure1, fileName);
            end
        end
        close(figure1);
        % scalingFactor = 30; % Adjust this value based on how spacious you want the plot
        % Set the figure size
        % figure('Position', [100, 100, settings.dimensions(1)*scalingFactor, settings.dimensions(2)*scalingFactor]); % [left, bottom, width, height]
        figure2 = figure;
        plotsomhits(selfOrgMap,data);
        set(gca, 'XTick', [], 'YTick', []);
        % Loop through each file format and save the figure

        for iFormat = 1:length(settings.storeFormats)
            % Create the file name with the desired format
            fileName = fullfile(dirName, ['Hit Histogram' settings.storeFormats{iFormat}]);

            % Save the figure in the specified format
            if strcmp(settings.storeFormats{iFormat}, '.fig')
                % For .fig, use savefig
                savefig(fileName);
            else
                % For other formats, use print
                exportgraphics(figure2, fileName);
            end
        end
        close(figure2);
        %% Extract the weights
        ws = selfOrgMap.IW{1, 1}';
        fileName = fullfile(baseDir, 'output.hdf5');
        
        h5create(fileName, "/"+fieldNames{iName}+ "/weights", size(ws));
        h5write(fileName, "/"+fieldNames{iName}+'/weights', ws)

        %% apply SOM to each dataset
        fieldNames = fieldnames(dataStruct);
        x = data;
        y = selfOrgMap(x);
        clusterIndex = vec2ind(y)';
        histmp = histcounts(clusterIndex, length(clusterIndex), 'BinMethod','integers')';
        h5create(fileName, "/"+fieldNames{iName}+"/hitHistorgam" , size(histmp));
        h5write(fileName, "/"+fieldNames{iName}+"/hitHistorgam", histmp);
        
        h5create(fileName, "/"+fieldNames{iName}+"/clusterIndex", size(clusterIndex));
        h5write(fileName, "/"+fieldNames{iName}+"/clusterIndex", clusterIndex);

    end

end