I will briefly discuss all the files contained in the present folder.

```train.txt``` : list of all the training images of the Pascal VOC 2012 dataset   
```val.txt``` : list of all the validation images of the Pascal VOC 2012 dataset


```train_1_19.txt``` : list of the training images of the Pascal VOC 2012 dataset containing the classes from 0 to 19th included   
```val_1_19.txt``` : list of the validation images of the Pascal VOC 2012 dataset containing the classes from 0 to 19th included


```train_20.txt``` : list of the training images of the Pascal VOC 2012 dataset containing the classes from 20th to 20th included (images that will be used for the incremental step, may contain previously seen classes)

```val_1_20.txt``` : list of the validation images of the Pascal VOC 2012 dataset containing the classes from 0th to 20th included

```divide_dataset_1_to_x.m``` : utility function to divide the dataset   
```divide_dataset_x_to_y.m``` : utility function to divide the dataset


```build_data.py``` : script to generate .tfrecords files containing images to be used for training   
```build_data.bat``` : sample bat file for Windows to call the build_data.py function



The following folders (currently empty) have to be filled with:
```JPEGImages\``` : folder containing all the RGB images of our dataset [to be downloaded from the Pascal VOC 2012 webpage]  
```SegmentationClassAug\``` : folder containing all the annotations of our dataset [to be downloaded from the Pascal VOC 2012 webpage]   
```SegmentationClassAug_color\``` : folder containing all the annotations of our dataset in color    
```tfrecords\``` : folder containing our tfrecords file to be used by TensorFlow dor training [to be downloaded from our paper webpage]   