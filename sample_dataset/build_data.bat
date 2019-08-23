call python -u build_data.py ^
--data_list train_1_19.txt ^
--output_file tfrecords/train_1_19.tfrecords

call python -u build_data.py ^
--data_list val_1_19.txt ^
--output_file tfrecords/val_1_19.tfrecords

call python -u build_data.py ^
--data_list train_20.txt ^
--output_file tfrecords/train_20.tfrecords

call python -u build_data.py ^
--data_list val_1_20.txt ^
--output_file tfrecords/val_1_20.tfrecords
