% Baseline training
clear; close all;

cpathTr = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LR\Lesion\lesion_cycle_gan\test_25\images\*real.png';
npathTr = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LS\Normal\lesion_cycle_gan\test_25\images\*real.png';

cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
cTr.Labels = repmat(categorical("Lesion"),[length(cTr.Labels),1]);
nTr = imageDatastore(npathTr,'LabelSource','foldernames');
nTr.Labels = repmat(categorical("Normal"),[length(nTr.Labels),1]);

TrainPt = cat(1,nTr.Files,cTr.Files);
TrainLb = cat(1,nTr.Labels,cTr.Labels);

TrainData = imageDatastore(TrainPt,"Labels",TrainLb);

[TrainData,ValData] = splitEachLabel(TrainData,0.8,0.2);

augmenter = imageDataAugmenter( ...
    'RandRotation',[-30 30],...
    'RandXReflection',1,...
    'RandYReflection',1,...
    'RandXScale',[0.5 1.5],...
    'RandYScale',[0.5 1.5]);

AugTrainData = augmentedImageDatastore([224 224 3],TrainData,"DataAugmentation",augmenter);
AugValData = augmentedImageDatastore([224 224 3],ValData);
%
net = resnet18;
% net = efficientnetb0;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
% lgraph = removeLayers(lgraph,{'efficientnet-b0|model|head|dense|MatMul','Softmax','classification'});
%     lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});

newLayers = [
%         fullyConnectedLayer(100,'Name','fc100','WeightLearnRateFactor',10,'BiasLearnRateFactor', 10)
    fullyConnectedLayer(2,'Name','fc2','WeightLearnRateFactor',1,'BiasLearnRateFactor', 1)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5','fc2');
% lgraph = connectLayers(lgraph,'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool','fc2');
%   lgraph = connectLayers(lgraph,'avg_pool','fc2');
layers = lgraph;

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
save resnet18_baseline_LR_LS3.mat net lgraph
% save efficientnetb0_baseline_LR_LS.mat net lgraph
%% Baseline testing
load resnet18_baseline_LR_LS3.mat
% load efficientnetb0_baseline_LR_LS.mat net lgraph
cpathT = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LR_heldout\Lesion\lesion_cycle_gan\test_25\images\*CC*real.png';
npathT = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LS_heldout\Normal\lesion_cycle_gan\test_25\images\*CC*real.png';

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
histogram(probs(TestLb=="Lesion",2))
hold on;
histogram(probs(TestLb=="Normal",2))


[YPred,probs] = classify(net,AugTrainData);
[x,y,~,auc_tr] = perfcurve(TrainData.Labels,probs(:,2),"Lesion");
disp(auc_tr);
figure(2);
subplot(131); plot(x,y);
subplot(132); 
histogram(probs(TrainData.Labels=="Lesion",2))
hold on;
histogram(probs(TrainData.Labels=="Normal",2))

%% Baseline testing
load resnet18_baseline_LR_LS.mat
% load efficientnetb0_baseline_LR_LS.mat net lgraph
cpathT = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LR_heldout\Lesion\lesion_cycle_gan\test_25\images\*CC*real.png';
npathT = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LS_heldout\Normal\lesion_cycle_gan\test_25\images\*CC*real.png';

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
subplot(221); plot(x,y);
subplot(222); 
histogram(probs(TestLb=="Lesion",2))
hold on;
histogram(probs(TestLb=="Normal",2))

Th = 0.98;
SindC = (probs(:,2)<Th)&(TestLb=="Lesion");
SindN = (probs(:,2)>(1-Th))&(TestLb=="Normal");

TestPt3 = cat(1,TestPt(SindC),TestPt(SindN));
TestLb3 = cat(1,TestLb(SindC),TestLb(SindN));

TestData3 = imageDatastore(TestPt3,"Labels",TestLb3);

AugTestData = augmentedImageDatastore([224 224 3],TestData3);
[YPred,probs2] = classify(net,AugTestData);
[x,y,~,auc_t] = perfcurve(TestLb3,probs2(:,2),"Lesion");
disp(auc_t);

subplot(223); plot(x,y);
subplot(224); 
histogram(probs2(TestLb3=="Lesion",2))
hold on;
histogram(probs2(TestLb3=="Normal",2))

[YPred,probs] = classify(net,AugTrainData);
[x,y,~,auc_tr] = perfcurve(TrainData.Labels,probs(:,2),"Lesion");
disp(auc_tr);
figure(2);
subplot(131); plot(x,y);
subplot(132); 
histogram(probs(TrainData.Labels=="Lesion",2))
hold on;
histogram(probs(TrainData.Labels=="Normal",2))
% find easy cases
% Threshold 0.5?
%%
% easy caes
% Th = 0.99;
% indC = (probs(:,2)>Th)&(TrainData.Labels=="Lesion");
% indN = (probs(:,2)<(1-Th))&(TrainData.Labels=="Normal");
%difficult cases
Th = 0.25;
indC = (probs(:,2)<Th)&(TrainData.Labels=="Lesion");
indN = (probs(:,2)>(1-Th))&(TrainData.Labels=="Normal");

subplot(133); 
histogram(probs(indC,2))
hold on;
histogram(probs(indN,2))
%
cpathTr = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LR\Lesion\lesion_cycle_gan\test_latest\images\*CC*fake.png';
npathTr = 'K:\Lesion_normal_cycleGAN_Dec2023\data_proc_LS\Normal\lesion_cycle_gan\test_latest\images\*CC*fake.png';

cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
cTr.Labels = repmat(categorical("Lesion"),[length(cTr.Labels),1]);
nTr = imageDatastore(npathTr,'LabelSource','foldernames');
nTr.Labels = repmat(categorical("Normal"),[length(nTr.Labels),1]);

TrainPt = cat(1,nTr.Files,cTr.Files);
TrainLb = cat(1,nTr.Labels,cTr.Labels);

TrainData2 = imageDatastore(TrainPt,"Labels",TrainLb);

[TrainData2,ValData2] = splitEachLabel(TrainData2,0.8,0.2);

TrainPt3 = cat(1,TrainData2.Files(indC|indN),TrainData.Files(~(indC|indN)));
TrainLb3 = cat(1,TrainData2.Labels(indC|indN),TrainData.Labels(~(indC|indN)));

TrainData3 = imageDatastore(TrainPt3,"Labels",TrainLb3);

AugTrainData3 = augmentedImageDatastore([224 224 3],TrainData3,"DataAugmentation",augmenter);
layers = lgraph;
%
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

net2 = trainNetwork(AugTrainData3,layers,options);
save(['resnet18_LR_LS_TF_ep100_thr',num2str(Th),'_hardc.mat'],'net2');

% Test the result
load resnet18_baseline_LR_LS.mat
% load efficientnetb0_baseline_LR_LS.mat
TestData3 = imageDatastore(TestPt3,"Labels",TestLb3);

AugTestData = augmentedImageDatastore([224 224 3],TestData3);

[YPred,probs] = classify(net,AugTestData);
[x,y,~,auc_t] = perfcurve(TestLb3,probs(:,2),"Lesion");
disp(auc_t);

ldtxt = 'resnet18_LR_LS_TF_ep100_thr0.25_hardc';
% ldtxt = 'efficientnetb0_LR_LS_TF_ep50_thr0.5_hardc'; 

load([ldtxt,'.mat']);
[YPred,probs] = classify(net2,AugTestData);
[x,y,~,auc_t_aug] = perfcurve(TestLb3,probs(:,2),"Lesion");
figure(3);
subplot(121); plot(x,y);
subplot(122); 
histogram(probs(TestLb3=="Lesion",2))
hold on;
histogram(probs(TestLb3=="Normal",2))


% [YPred,probs] = classify(net2,AugTrainData3);
% [x,y,~,auc_tr_aug] = perfcurve(TrainLb3,probs(:,2),"Lesion");
% figure(4);
% subplot(121); plot(x,y);
% subplot(122); 
% histogram(probs(TrainLb3=="Lesion",2))
% hold on;
% histogram(probs(TrainLb3=="Normal",2))

% save('resnet18_LR_LS_hardc_AUCs.mat','auc_t','auc_t_aug');
save([ldtxt,'_AUCs.mat'],'auc_t','auc_t_aug');