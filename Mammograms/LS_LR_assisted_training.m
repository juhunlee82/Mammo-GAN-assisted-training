%% LS LR assisted training
% run Baseline_training.m first
% load baseline
load resnet18_baseline_LR_LS.mat

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

[YPred,probs] = classify(net,AugTrainData);
[x,y,~,auc_tr] = perfcurve(TrainData.Labels,probs(:,2),"Lesion");

% Threshold values for identifying easy and difficult cases
Thlist = [0.05, 0.1, 0.25, 0.5 0.75 1];
% Epochs to consider
list = {'25','50','75','latest'};

for j = 1:6
    Th = Thlist(j);
    indC = (probs(:,2)<Th)&(TrainData.Labels=="Lesion");
    indN = (probs(:,2)>(1-Th))&(TrainData.Labels=="Normal");

    for i = 1:4
        cpathTr = ['\Lesion\lesion_cycle_gan\test_',list{i},'\images\*fake.png'];
        npathTr = ['\Normal\lesion_cycle_gan\test_',list{i},'\images\*fake.png'];


        cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
        cTr.Labels = repmat(categorical("Lesion"),[length(cTr.Labels),1]);
        nTr = imageDatastore(npathTr,'LabelSource','foldernames');
        nTr.Labels = repmat(categorical("Normal"),[length(nTr.Labels),1]);

        TrainPt = cat(1,nTr.Files,cTr.Files);
        TrainLb = cat(1,nTr.Labels,cTr.Labels);
        
        TrainData2 = imageDatastore(TrainPt,"Labels",TrainLb);
        
        [TrainData2,ValData2] = splitEachLabel(TrainData2,0.8,0.2);

        % replace easy samples with LS LR converted ones
        TrainPt3 = cat(1,TrainData2.Files(indC|indN),TrainData.Files);
        TrainLb3 = cat(1,TrainData2.Labels(indC|indN),TrainData.Labels);

        TrainData3 = imageDatastore(TrainPt3,"Labels",TrainLb3);

        AugTrainData3 = augmentedImageDatastore([224 224 3],TrainData3,"DataAugmentation",augmenter);
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
            'Verbose',true, ...
            'Plots','none',...
            'OutputNetwork','best-validation-loss');
        
        net2 = trainNetwork(AugTrainData3,layers,options);
        save(['resnet18_LR_LS_TF_ep',list{i},'_thr',num2str(Th),'.mat'],'net2');

    end
end
