%% Baseline training
clear; close all;
% path for development dataset. cpathTr for lesion case, npathTr for normal case 
% specify your path to original patch dataset
cpathTr = '\lesion\*.png';
npathTr = '\normal\*.png';

% assign labels
cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
cTr.Labels = repmat(categorical("Lesion"),[length(cTr.Labels),1]);
nTr = imageDatastore(npathTr,'LabelSource','foldernames');
nTr.Labels = repmat(categorical("Normal"),[length(nTr.Labels),1]);

TrainPt = cat(1,nTr.Files,cTr.Files);
TrainLb = cat(1,nTr.Labels,cTr.Labels);

TrainData = imageDatastore(TrainPt,"Labels",TrainLb);

% divide development dataset into train, val
[TrainData,ValData] = splitEachLabel(TrainData,0.8,0.2);

% data augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation',[-30 30],...
    'RandXReflection',1,...
    'RandYReflection',1,...
    'RandXScale',[0.5 1.5],...
    'RandYScale',[0.5 1.5]);

AugTrainData = augmentedImageDatastore([224 224 3],TrainData,"DataAugmentation",augmenter);
AugValData = augmentedImageDatastore([224 224 3],ValData);
% load resnet18
net = resnet18;

% replace last few layers for fine-tuning
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(2,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor', 1)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5','fc2');

layers = lgraph;

% training options
miniBatchSize = 32;
valFrequency = floor(numel(AugTrainData.Files)/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',AugValData, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(AugTrainData,layers,options);
% save baseline network
save resnet18_baseline_LR_LS.mat net lgraph

%% Baseline testing
load resnet18_baseline_LR_LS.mat

% path for testing dataset. cpathTr for lesion case, npathTr for normal case 
% specify your path to original patch dataset
cpathTr = '\lesion_test\*.png';
npathTr = '\normal_test\*.png';

cT = imageDatastore(cpathT,'LabelSource','foldernames');
cT.Labels = repmat(categorical("Lesion"),[length(cT.Labels),1]);
nT = imageDatastore(npathT,'LabelSource','foldernames');
nT.Labels = repmat(categorical("Normal"),[length(nT.Labels),1]);

TestPt = cat(1,nT.Files,cT.Files);
TestLb = cat(1,nT.Labels,cT.Labels);

TestData = imageDatastore(TestPt,"Labels",TestLb);
AugTestData = augmentedImageDatastore([224 224 3],TestData);

[YPred,probs] = classify(net,AugTestData);
[x,y,~,auc_t] = perfcurve(TestLb,probs(:,2),"Lesion");

disp(auc_t);
figure(1);
subplot(121); plot(x,y);
subplot(122); 
histogram(probs(TestLb=="Lesion",2));
hold on;
histogram(probs(TestLb=="Normal",2));
