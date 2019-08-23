% script to divide the dataset from first class to the class_number class
clc; clear all; close all;
type = {'train','val'};

for type_idx=1:length(type)
    for class_number=[19]
        mkdir(['incremental_images/',type{type_idx},'_img_1_',num2str(class_number)])
        disp([type{type_idx}, num2str(class_number)])
        load(['matrix_classes_',type{type_idx},'.mat']);
        matrix_classes_summed=[];
        matrix_classes_summed = sum(matrix_classes(:,class_number+1:size(matrix_classes,2)),2);
        
        fileID = fopen([type{type_idx},'_1_',num2str(class_number),'.txt'],'w+');
        for i=1:length(matrix_classes_summed)
            if matrix_classes_summed(i)==0
                fprintf(fileID, [img_gt_list{1,1}{i,1}, ' ', img_gt_list{1,2}{i,1}, '\n']);
                copyfile(['SegmentationClassAug_color/',extractAfter(img_gt_list{1,2}{i,1},22)],...
                    ['incremental_images/',type{type_idx},'_img_1_', num2str(class_number),'/', extractAfter(img_gt_list{1,2}{i,1},22)]);
            end
        end
        fclose(fileID);   
    end
end


%% Divide only the JPEG Images
clc; clear all; close all;
type = {'train','val'};

for type_idx=1:length(type)
    for class_number=[19]
        mkdir(['incremental_JPEGimages/',type{type_idx},'_img_1_',num2str(class_number)])
        disp([type{type_idx}, num2str(class_number)])
        load(['matrix_classes_',type{type_idx},'.mat']);
        matrix_classes_summed=[];
        matrix_classes_summed = sum(matrix_classes(:,class_number+1:size(matrix_classes,2)),2);
       
        for i=1:length(matrix_classes_summed)
            if matrix_classes_summed(i)==0
                copyfile(['JPEGImages/',extractAfter(img_gt_list{1,1}{i,1},12)],...
                    ['incremental_JPEGimages/',type{type_idx},'_img_1_', num2str(class_number),'/', extractAfter(img_gt_list{1,1}{i,1},12)]);
            end
        end  
    end
end

