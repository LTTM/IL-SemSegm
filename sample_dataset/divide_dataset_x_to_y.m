% script to divide the dataset from from_x to the to_y class
clc; clear all; close all;
type = {'train','val'};
from_x=20;
to_y=20;

for type_idx=1:length(type)
    mkdir(['incremental_images/',type{type_idx},'_img_',num2str(from_x),'_',num2str(to_y)])
    disp([type{type_idx}, num2str(from_x),'_',num2str(to_y)])
    load(['matrix_classes_',type{type_idx},'.mat']);
    matrix_classes_summed=[];
    matrix_classes_summed = sum(matrix_classes(:,from_x:to_y),2);
    
    fileID = fopen([type{type_idx},'_',num2str(from_x),'_',num2str(to_y),'.txt'],'w+');
    for i=1:length(matrix_classes_summed)
        if matrix_classes_summed(i)~=0
            fprintf(fileID, [img_gt_list{1,1}{i,1}, ' ', img_gt_list{1,2}{i,1}, '\n']);
            copyfile(['SegmentationClassAug_color/',extractAfter(img_gt_list{1,2}{i,1},22)],...
                ['incremental_images/',type{type_idx},'_img_',num2str(from_x),'_',num2str(to_y),'/', extractAfter(img_gt_list{1,2}{i,1},22)]);
        end
    end
    fclose(fileID);
end


%% %% Divide only the JPEG Images
clc; clear all; close all;
type = {'train','val'};
from_x=20;
to_y=20;

for type_idx=1:length(type)
    mkdir(['incremental_JPEGimages/',type{type_idx},'_img_',num2str(from_x),'_',num2str(to_y)])
    disp([type{type_idx}, num2str(from_x),'_',num2str(to_y)])
    load(['matrix_classes_',type{type_idx},'.mat']);
    matrix_classes_summed=[];
    matrix_classes_summed = sum(matrix_classes(:,from_x:to_y),2);
    
    for i=1:length(matrix_classes_summed)
        if matrix_classes_summed(i)~=0
            copyfile(['JPEGImages/',extractAfter(img_gt_list{1,1}{i,1},12)],...
                ['incremental_JPEGimages/',type{type_idx},'_img_',num2str(from_x),'_',num2str(to_y),'/', extractAfter(img_gt_list{1,1}{i,1},12)]);
        end
    end
end



