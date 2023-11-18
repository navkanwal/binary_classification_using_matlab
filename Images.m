clear all
clc

dataDir = 'D:\machine_Learning\assignment2\data';

% Create imageDatastore for cat
catDS = imageDatastore(fullfile(dataDir, 'cat'), 'LabelSource', 'foldernames');
catDS.Labels = zeros(size(catDS.Labels));
% converting images to RGB
imageSize = [64,64];
numcatImages = numel(catDS.Files);
catRGB = [];
for i = 1 : numcatImages
    currImagePath = catDS.Files{i};
    currImageMatrix = imread(currImagePath);
    resizedImage = imresize(currImageMatrix,imageSize);
    linearVector = reshape(resizedImage, 1, []);
    catRGB = [catRGB;linearVector];    
end
catRGB = double(catRGB)/255;

rabbitDS = imageDatastore(fullfile(dataDir,'rabbit'), 'LabelSource', 'foldernames');
numrabbitImages = numel(rabbitDS.Files);
rabbitRGB = [];
for i = 1 : numrabbitImages
    currImagePath = rabbitDS.Files{i};
    currImageMatrix = imread(currImagePath);
    if size(currImageMatrix, 3) == 4
        continue;  % Skip this image and proceed to the next one
    end
    resizedImage = imresize(currImageMatrix,imageSize);
    linearVector = reshape(resizedImage, 1, []);
    
    rabbitRGB = [rabbitRGB;linearVector]; 
end
rabbitRGB = double(rabbitRGB)/255;

%Defining Class of Each
%cat - 0
%rabbit - 1

% Add a column of zeros to 'catRGB'
catRGB = [catRGB, zeros(size(catRGB, 1), 1)];

% Add a column of ones to 'rabbitRGB'
rabbitRGB = [rabbitRGB, ones(size(rabbitRGB, 1), 1)];

dataRGB = [catRGB;rabbitRGB];


numRows = size(dataRGB, 1);
randomIndices = randperm(numRows);


randomizedDataRGB = dataRGB(randomIndices, :);


% Define the proportions for the split
trainPercent = 0.60;  % 60% for training
validationPercent = 0.20;  % 20% for validation
testPercent = 0.20;  % 20% for testing

% Split the data
numRows = size(randomizedDataRGB, 1);
numTrain = round(trainPercent * numRows);
numValidation = round(validationPercent * numRows);

trainData = randomizedDataRGB(1:numTrain, :);
validationData = randomizedDataRGB(numTrain + 1:numTrain + numValidation, :);
testData = randomizedDataRGB(numTrain + numValidation + 1:end, :);

X = trainData(:, 1:12288);
y = trainData(:, 12289);

[m , n] = size(X);

X = [ones(m , 1) X];
initial_theta = zeros(n+1,1);

%Compute Cost and Display Cost and Gradient
[cost,grad] = costFunction(initial_theta,X,y);



options = optimset('GradObj', 'on', 'MaxIter', 400);


[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);



%Confustion Matrix
%m is the number of images in X 
predictedlabels = [];

for i = 1:m
    currvector = X(i , :);
    prob = predict(theta,currvector);
    predictedlabels = [predictedlabels;prob];    
end

Xtest = testData(:, 1:12288);
[mtest,ntest] = size(Xtest);
Xtest = [ones(mtest , 1) Xtest];

ytest = testData(:, 12289);
predictedlabelstest = [];
for i = 1:mtest
    currvector = Xtest(i , :);
    prob = predict(theta,currvector);
    predictedlabelstest = [predictedlabelstest;prob];    
end
C = confusionmat(ytest,predictedlabelstest);
fprintf('The Confusion Matrix for RGB Model is C = \n');
disp(C);
precision = C(1,1)/(C(1,1)+C(2,1));
recall = C(1,1)/(C(1,1)+C(1,2));
f1score = (2*precision*recall)/(precision+recall);
fprintf('The F1 Score of the RGB Model is %d.\n',f1score);



