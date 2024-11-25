clear; close all;

load resnet18_baseline_chest.mat
cpathTr = '\lesions\*.png';
npathTr = '\negative\*.png';

cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
cTr.Labels = repmat(categorical("Lesion"),[length(cTr.Labels),1]);
nTr = imageDatastore(npathTr,'LabelSource','foldernames');
nTr.Labels = repmat(categorical("Normal"),[length(nTr.Labels),1]);

TrainPt = cat(1,nTr.Files,cTr.Files);
TrainLb = cat(1,nTr.Labels,cTr.Labels);

TrainData = imageDatastore(TrainPt,"Labels",TrainLb);

[TrainData,ValData,TestData] = splitEachLabel(TrainData,0.7,0.1,0.2);

augmenter = imageDataAugmenter( ...
    'RandRotation',[-30 30],...
    'RandXReflection',1,...
    'RandYReflection',1,...
    'RandXScale',[0.5 1.5],...
    'RandYScale',[0.5 1.5]);

AugTrainData = augmentedImageDatastore([224 224 3],TrainData,"DataAugmentation",augmenter);
AugValData = augmentedImageDatastore([224 224 3],ValData);
AugTestData = augmentedImageDatastore([224 224 3],TestData);

% Th = 0.5 0.75 and Epoch 50 75 works for mammograms. So we used the same for Chest X-ray
Thlist = [0.5 0.75];
list = {'50','75'};

for j = 1:2
    Th = Thlist(j);
    % finding target training cases for LS - LR process to increase its difficult level. 
    indC = (probs(:,2)<Th)&(TrainData.Labels=="Lesion");
    indN = (probs(:,2)>(1-Th))&(TrainData.Labels=="Normal");

    for i = 1:2
    % the following paths include LS LR converted cases for the development set. 'fake' indicates they are LS LR converted.
        cpathTr = ['\Lesion\lesion_cycle_gan\test_',list{i},'\images\*fake.png'];
        npathTr = ['\Normal\lesion_cycle_gan\test_',list{i},'\images\*fake.png'];

        cTr = imageDatastore(cpathTr,'LabelSource','foldernames');
        cTr.Labels = repmat(categorical("Lesion"),[length(cTr.Labels),1]);
        nTr = imageDatastore(npathTr,'LabelSource','foldernames');
        nTr.Labels = repmat(categorical("Normal"),[length(nTr.Labels),1]);

        TrainPt = cat(1,nTr.Files,cTr.Files);
        TrainLb = cat(1,nTr.Labels,cTr.Labels);

        TrainData2 = imageDatastore(TrainPt,"Labels",TrainLb);
        
        [TrainData2,ValData2,TestData2] = splitEachLabel(TrainData2,0.7,0.1,0.2);

        % replace easy samples with LS LR converted ones
        % TrainData2 include LS LR converted of TrainData
        TrainPt3 = cat(1,TrainData2.Files(indC|indN),TrainData.Files(~(indC|indN)));
        TrainLb3 = cat(1,TrainData2.Labels(indC|indN),TrainData.Labels(~(indC|indN)));

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
