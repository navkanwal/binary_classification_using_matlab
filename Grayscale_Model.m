clear all
clc

dataDir = 'D:\machine_Learning\assignment2\data';


catDS = imageDatastore(fullfile(dataDir, 'cat'), 'LabelSource', 'foldernames');
catDS.Labels = zeros(size(catDS.Labels));

imageSize = [64,64];
numcatImages = numel(catDS.Files);
catGrayScale = [];
for i = 1 : numcatImages
    currImagePath = catDS.Files{i};
    currImageMatrix = imread(currImagePath);
    currImageMatrix = rgb2gray(currImageMatrix);
    resizedImage = imresize(currImageMatrix,imageSize);
    linearVector = reshape(resizedImage, 1, []);
    catGrayScale = [catGrayScale;linearVector];    
end
catGrayScale = double(catGrayScale)/255;

rabbitDS = imageDatastore(fullfile(dataDir,'rabbit'), 'LabelSource', 'foldernames');
numrabbitImages = numel(rabbitDS.Files);
rabbitGrayScale = [];
for i = 1 : numrabbitImages
    currImagePath = rabbitDS.Files{i};
    currImageMatrix = imread(currImagePath);
    currImageMatrix = rgb2gray(currImageMatrix);
    resizedImage = imresize(currImageMatrix,imageSize);
    linearVector = reshape(resizedImage, 1, []);
    
    rabbitGrayScale = [rabbitGrayScale;linearVector]; 
end
rabbitGrayScale = double(rabbitGrayScale)/255;


catGrayScale = [catGrayScale, zeros(size(catGrayScale, 1), 1)];


rabbitGrayScale = [rabbitGrayScale, ones(size(rabbitGrayScale, 1), 1)];

dataGrayScale = [catGrayScale;rabbitGrayScale];
x

numRows = size(dataGrayScale, 1);
randomIndices = randperm(numRows);

% Rearrange the rows of the 'dataGrayScale' matrix
randomizedDataGrayScale = dataGrayScale(randomIndices, :);



% Define the proportions for the split
trainPercent = 0.60;  % 60% for training
validationPercent = 0.20;  
testPercent = 0.20;  

% Split the data
numRows = size(randomizedDataGrayScale, 1);
numTrain = round(trainPercent * numRows);
numValidation = round(validationPercent * numRows);

trainData = randomizedDataGrayScale(1:numTrain, :);
validationData = randomizedDataGrayScale(numTrain + 1:numTrain + numValidation, :);
testData = randomizedDataGrayScale(numTrain + numValidation + 1:end, :);

X = trainData(:, 1:4096);
y = trainData(:, 4097);

[m , n] = size(X);

X = [ones(m , 1) X];
initial_theta = zeros(n+1,1);

%Compute Cost and Display Cost and Gradient
[cost,grad] = costFunction(initial_theta,X,y);

%fminunc part
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
%fminunc part ends


%Confustion Matrix
%m is the number of images in X 
predictedlabels = [];

for i = 1:m
    currvector = X(i , :);
    prob = predict(theta,currvector);
    predictedlabels = [predictedlabels;prob];    
end

Xtest = testData(:, 1:4096);
[mtest,ntest] = size(Xtest);
Xtest = [ones(mtest , 1) Xtest];

ytest = testData(:, 4097);
predictedlabelstest = [];
for i = 1:mtest
    currvector = Xtest(i , :);
    prob = predict(theta,currvector);
    predictedlabelstest = [predictedlabelstest;prob];    
end

C = confusionmat(ytest,predictedlabelstest);
fprintf('The Confusion Matrix for GrayScale Model is C = \n');
disp(C);
precision = C(1,1)/(C(1,1)+C(2,1));
recall = C(1,1)/(C(1,1)+C(1,2));
fscore = (2*precision*recall)/(precision+recall);
fprintf('The F1 Score of the GrayScale Model is %d.\n', fscore);





