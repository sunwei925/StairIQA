CUDA_VISIBLE_DEVICES=0 python -u train_single_database.py \
--num_epochs 100 \
--batch_size 30 \
--resize 384 \
--crop_size 320 \
--lr 0.00005 \
--decay_ratio 0.9 \
--decay_interval 10 \
--snapshot /data/sunwei_data/ModelFolder/StairIQA/ \
--database_dir /data/sunwei_data/BID/ImageDatabase/ImageDatabase/ \
--model stairIQA_resnet \
--multi_gpu False \
--print_samples 20 \
--database BID \
--test_method five \
>> logfiles/train_BID_stairIQA_resnet.log




CUDA_VISIBLE_DEVICES=0 python -u train_imdt.py \
--num_epochs 3 \
--batch_size 30 \
--lr 0.00001 \
--decay_ratio 0.9 \
--decay_interval 1 \
--snapshot /data/sunwei_data/ModelFolder/StairIQA/ \
--model stairIQA_resnet \
--multi_gpu False \
--print_samples 100 \
--test_method five \
--results_path results \
--exp_id 0 \
>> logfiles/train_stairIQA_resnet_imdt_exp_id_0.log

