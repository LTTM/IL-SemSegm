call python -u train_with_distillation.py ^
--batch_size 4 ^
--training_steps 20000 ^
--start_decay_step 0 ^
--validation_batch_size 1449 ^
--logging_steps_interval 100 ^
--validation_steps_interval 1000 ^
--save_steps_interval 1000 ^
--decay_steps 20000 ^
--start_learning_rate 1e-4 ^
--end_learning_rate 1e-6 ^
--training_file ../../sample_dataset/tfrecords/train_1_19.tfrecords ^
--validation_file ../../sample_dataset/tfrecords/val_1_19.tfrecords ^
--up_to_class 19 ^
--lambda_distillation 0 ^
--is_incremental 0 ^
--is_save_new_softmax_maps 0 ^
1> output_1_19.txt 2>&1

call python run_generate_old_maps.py ^
--load_model 1_to_19 ^
--validation_file ../../sample_dataset/tfrecords/train_20.tfrecords

call python -u train_with_distillation.py ^
--batch_size 4 ^
--training_steps 21000 ^
--start_decay_step 20000 ^
--validation_batch_size 1449 ^
--logging_steps_interval 100 ^
--validation_steps_interval 1000 ^
--save_steps_interval 1000 ^
--decay_steps 20000 ^
--start_learning_rate 5e-5 ^
--end_learning_rate 1e-6 ^
--load_model 1_to_19 ^
--training_file ../../sample_dataset/tfrecords/train_20.tfrecords ^
--validation_file ../../sample_dataset/tfrecords/val_1_20.tfrecords ^
--from_new_class 20 ^
--to_new_class 20 ^
--standard_loss_applied_to all ^
--distill_loss_applied_to old_bgr ^
--lambda_distillation 0 ^
--is_incremental 1 ^
--is_save_new_softmax_maps 0 ^
1> output_1_19_20_nodist.txt 2>&1

call python -u train_with_distillation.py ^
--batch_size 4 ^
--training_steps 21000 ^
--start_decay_step 20000 ^
--validation_batch_size 1449 ^
--logging_steps_interval 100 ^
--validation_steps_interval 1000 ^
--save_steps_interval 1000 ^
--decay_steps 20000 ^
--start_learning_rate 5e-5 ^
--end_learning_rate 1e-6 ^
--load_model 1_to_19 ^
--training_file ../../sample_dataset/tfrecords/train_20.tfrecords ^
--validation_file ../../sample_dataset/tfrecords/val_1_20.tfrecords ^
--from_new_class 20 ^
--to_new_class 20 ^
--standard_loss_applied_to all ^
--distill_loss_applied_to old_bgr ^
--lambda_distillation 1 ^
--is_incremental 1 ^
--is_save_new_softmax_maps 0 ^
1> output_1_19_20_LD1.txt 2>&1
