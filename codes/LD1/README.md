I will briefly discuss in alphabetical order all the files contained in the present folder.

```export_graph.py``` : freeze variables and convert the segmentation networks to a GraphDef file.

```inference.py``` : inference step over the saved checkpoint

```model.py``` : model of the graph

```model_with_distillation.py``` : model of the graph with the distillation loss

```network.py``` : model of the Deeplab-v2 network

```paper_results/output_1_19.txt``` : output of the first step of training on the first 20 classes.

```paper_results/output_1_19_20_LD1.txt``` : output of the incremental step where we load the model saved at the end of 'output_1_19.txt' and we try to learn the last class with LD1 as distillation loss

```paper_results/output_1_19_20_nodist``` : output of the incremental step where we load the model saved at the end of 'output_1_19.txt' and we try to learn the last class

```reader.py``` : reader class to read the tfrecords files

```run_generate_old_maps.py``` :  after the first stage of training this code will generate the necessary files for the incremental step. Namely it will create three folders containing the input images and the respective input labels for the next step, and the output of the softmax at the current step on which we want to apply the distillation loss at the next step.  
**NB:** We have verified that doing the procedure on the fly (i.e. not saving anything to disk) or not does not change the results we present here the second version since it is more stable and could be run in two steps. Notice that this is only for convenience and the results are completely not affected (difference of mIoU <0.05%) by the procedure according to which the inference over the previous network is made during the incremental step.

```script.bat``` : this is the .bat file to run (in Windows environments) to achieve the whole training procedure. Check comments inside it.

```script.job``` :  this is the script to run (in Unix environments) to achieve the whole training procedure. Check comments inside it.

```test.py``` : file to test a model already trained on a validation or test set

```train_with_distillation.py``` : this is the main script you should run which performs the training procedures. Indeed, this is the file called by script.bat or script.job.

```utils/convert_rgb_2_labels.py``` : utils script to convert colors

```utils/image_utils.py``` : utility functions for image elaboration

```utils/tensor_utils.py``` : utility functions for tensor related operations





Umberto Michieli   
Ph.D. Student   
University of Padova   
Department of Information Engineering (DEI)   
Email: umberto.michieli@dei.unipd.it   
Website: https://umbertomichieli.github.io/   
