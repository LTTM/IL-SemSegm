The codes in this folder are for the scenario in which the last class is added to the model.
Compare with Table 1 of the paper:   
```Michieli U., Zanuttigh P., "Incremental Learning Techniques for Semantic Segmentation", Proceedings of the International Conference on Computer Vision (ICCV), Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV), Seoul (South Korea), 2 November 2019.```





```LD1/``` : the distillation loss LD1. Here there is also the result for the fine-tuning approach.


```LD2/``` : the distillation loss LD2.


```encoderfixed_LD1/``` : the distillation loss LD1 with the encoeder fixed.


```encoderfixed_LD2/``` : the distillation loss LD2 with the encoeder fixed.



NB: to have fair comparison be sure to run the first step only once and then copy/paste those checkpoints into the desired folder inside which you should only run the incremental step. 

