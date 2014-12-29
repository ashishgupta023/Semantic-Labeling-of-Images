%testSceneLabels - script used to test after SVM training is completed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up the environment to run SVM on data
%clear; clc; close all
%load 'svm_train_wspace';
categories = {'Sky','Tree','Road','Grass','Water','Bldg','Mtn','Fground'};
Nclasses = length(categories);
foldidx = 2;

%setup the features for testing
disp('Setting up the spix for the testing dataset')
test = test_idx{foldidx}; %NOTE to change this to match the training indexes
test_ids = zeros(size(F,1),1);
counter=0;
for n = 1:length(test)
  img_id = test(n);
  keys_img = keys(:,1);
  tmpids = find(keys_img == img_id);
  test_ids(counter+1:counter+size(tmpids,1))= tmpids;
  counter=counter+size(tmpids,1); 
end
test_ids = test_ids(1:counter);
Ftest = F(test_ids,:); %This as our testing data
Ctest = C(test_ids);
% Loop: compute the maximal score for features
disp('Using the SVM models to predict scores for test data')
scores = zeros(Nclasses, length(test_ids));
for c = 1:Nclasses
    disp(['Scores for class ' num2str(c)])
   %[~,score] = predict(netall{c},Ftest);
   %====LibSVM====%
   cd('libsvm-3.20\windows\');
   [~,accuracy , score] = svmpredict2(2*(Ctest==c)-1,Ftest,netall{c},'-b 1');
   cd('../..');
   %===Matlab SVMClassify%
   %[~,score] = svmclassify(netall{c}, Ftest);
    scores(c,:) = score(:,1);
end
whos scores netall Ftest

ctest_hat = zeros(length(test_ids), 1);
for k = 1:length(test_ids)
  [foo, ctest_hat(k)] = max(scores(:,k));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This section below is optional
% Plot performance
% Confusion matrix:
disp('Setting up the confusion matrix')
Cm = zeros(Nclasses, Nclasses);
for j = 1:Nclasses
    for i = 1:Nclasses
        % row i, col j is the percentage of images from class i that
        % were missclassified as class j.
        Cm(i,j) = 100*sum((C(test_ids)==i) .* (ctest_hat==j))/(0.0001+sum(C(test_ids)==i));
    end
end

figure
subplot(121)
imagesc(Cm); axis('square'); colorbar
subplot(122)
bar(diag(Cm))
title(mean(diag(Cm)))
axis('square')
