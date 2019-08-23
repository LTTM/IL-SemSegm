# Incremental Learning Techniques for Semantic Segmentation

This is the code of the paper: 
Michieli U., Zanuttigh P., "Incremental Learning Techniques for Semantic Segmentation", Proceedings of the International Conference on Computer Vision (ICCV), Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV), Seoul (South Korea), 2 November 2019.


If you find the code or the paper useful for your research please cite our work as:

```
@inproceedings{michieli2019,
  title={Incremental Learning Techniques for Semantic Segmentation},
  author={Michieli, Umberto and Zanuttigh, Pietro},
  booktitle={International Conference on Computer Vision (ICCV), Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV)},
  year={2019}
}
```

The PDF of the paper can be found at: https://arxiv.org/abs/1907.13372

The webpage of the paper is: https://lttm.dei.unipd.it/paper_data/IL/

# Setup
- Download the pretrained model of the Deeplab-v2 architecture pretrained on MSCOCO from: https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU and place it inside the folder ```pretrained/```
- Download the Pascal VOC dataset or your desired dataset and process it with the provided scripts for pre-processing (further info at ```sample_dataset/README.txt```).
	- For a quick start simply download the tfrecords folder provided from the webpage of the paper (https://lttm.dei.unipd.it/paper_data/IL/) and place it in ```sample_dataset/```
	- Alternatively, download the Pascal VOC 2012 and run ```sample_dataset/build_data.bat``` which will create 4 tfrecords files in ```sample_dataset/tfrecords/```


# Training
I would suggest to start from the folder ```codes/LD1/``` where you can run the simulations for the incremental scenario in which only the last class is added and only the LD1 loss is present.   
The procedure is discussed in detail at ```codes/LD1/README.txt```.   
**NB:** for the other folders no README.txt exists since they are straightforward extensions of the code contained in the ```LD1/``` folder


# Software & Hardware
- the code has been tested on Tensorlfow 1.10 - 1.14
- the vast majority of the experiments have been conducted on a computer with:
	- processor: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 3701 Mhz, 6 Core(s), 12 Logical Processor(s)
	- RAM: 32 GB
	- OS: Microsoft Windows 10 Pro Education
	- GPU: NVIDIA GeForce RTX 2080 Ti, 11GB


# Contact Information

I can provide checkpoints, datasets or other material upon specific request.

Umberto Michieli   
Ph.D. Student  
University of Padova  
Department of Information Engineering (DEI)  
Email: umberto.michieli@dei.unipd.it  
Personal Website: https://umbertomichieli.github.io/  

# License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
