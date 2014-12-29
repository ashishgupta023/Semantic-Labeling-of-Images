%runfile - script used to complete homework 4 using SVM classifier

% 1. Load the given image and superpixel data
load 'cvip_image_data';
imsegs=image_data;

% 2. Extract the features for all image superpixels and save workspace
% Then save the workspace as 'genfeatures.mat'
% Runs in 8-10 minutes
[features, labels, keys] = getSPixelFeatures(imsegs);
save genfeatures;

% 3. Train using the SVM classifier
% NOTE: In this script, you will need to change the SVM directory  
% setings and the cross-validation index number (between 1 and 5)
% This script saves the workspace 'svm_train_wspace.mat';
trainSceneLabels;
save svm_train_wspace;

% 4. Get the predictions made by the SVM classifier on test dataset
% Optionally, a confusion matrix can be generated to display results.
testSceneLabels;
save svm_test_wspace;