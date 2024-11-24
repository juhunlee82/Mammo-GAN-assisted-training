% Baseline training
clear; close all;

cpathTr = '\lesions\*.png';
npathTr = '\negative\*.png';

cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
nTr = imageDatastore(npathTr,'LabelSource','foldernames');

TrainPt = cat(1,nTr.Files,cTr.Files);
TrainLb = cat(1,nTr.Labels,cTr.Labels);

TrainData = imageDatastore(TrainPt,"Labels",TrainLb);
% divide dataset into 7:1:2 
[TrainData,ValData,TestData] = splitEachLabel(TrainData,0.7,0.1,0.2);

augmenter = imageDataAugmenter( ...
    'RandRotation',[-30 30],...
    'RandXReflection',1,...
    'RandYReflection',1,...
    'RandXScale',[0.5 1.5],...
    'RandYScale',[0.5 1.5]);

AugTrainData = augmentedImageDatastore([224 224 3],TrainData,"DataAugmentation",augmenter);
AugValData = augmentedImageDatastore([224 224 3],ValData);

net = resnet18;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(2,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor', 1)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5','fc2');
layers = lgraph;

miniBatchSize = 32;
valFrequency = floor(numel(AugTrainData.Files)/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',AugValData, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(AugTrainData,layers,options);
save resnet18_baseline_chest.mat net lgraph

%% Baseline testing
load resnet18_baseline_chest.mat

AugTestData = augmentedImageDatastore([224 224 3],TestData);

[YPred,probs] = classify(net,AugTestData);
[x,y,~,auc_t] = perfcurve(TestData.Labels,probs(:,2),"lesions");
disp(auc_t);
figure(1);
subplot(121); plot(x,y);
subplot(122); 
histogram(probs(TestData.Labels=="lesions",2))
hold on;
histogram(probs(TestData.Labels=="negative",2))

