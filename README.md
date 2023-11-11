# StairIQA
This is a repository for the models proposed in the paper "Blind Quality Assessment for in-the-Wild Images via Hierarchical Feature Fusion and Iterative Mixed Database Training" [JSTSP version](https://ieeexplore.ieee.org/abstract/document/10109108) [Arxiv version](https://arxiv.org/abs/2105.14550).

## Usage
### Download csv files
The train and test split files can be download from [Google drive](https://drive.google.com/file/d/121evqfjcsUwb014sOhl0mq7gMmaPuzpu/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/17zOm49cxZzhSqQCcsDv4jQ) (提取码：y4be)

### Train

Train on a single database (e.g. BID)
```
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
```

Train on multiple databases
```
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
```

The information of databases used in the train_imdt.py file can be edited in the config.yaml file.





### Test
Download the trained model:

Koniq10k: [Google drive](https://drive.google.com/file/d/1fKENSrGXao8po7R4yzK_YvxDxvzV4wTs/view?usp=sharing)

SPAQ: [Google drive](https://drive.google.com/file/d/1Px-PJE-08BPCfhP_gB7B78Z4Sp7dAi-4/view?usp=sharing)

BID: [Google drive](https://drive.google.com/file/d/1u6SfaXg1TaMDx7TmkKGfjOC163jmNg3J/view?usp=sharing)

LIVE_challenge: [Google drive](https://drive.google.com/file/d/1da4Aoe-zGvVljuB1PKOi4bnW2kBsnrP3/view?usp=sharing)

FLIVE: [Google drive](https://drive.google.com/file/d/14nV0R4AONnRD9EfgTD3uPVt_b1RgNZoN/view?usp=sharing)

FLIVE_patch: [Google drive](https://drive.google.com/file/d/1s5UyerDfjvGxE34OvMPOO7zBm_UO3BpZ/view?usp=sharing)

Test a image on the model where the regressor is trained on one dataset (i.e. Koniq10k):
```
CUDA_VISIBLE_DEVICES=0 python -u test_staircase.py \
--test_image_name image_name \
--model_path model_file \
--trained_database Koniq10k \
--test_method five \
--output_name output.txt
```


Test a image on the ensemble model:
```
CUDA_VISIBLE_DEVICES=1 python -u test_staircase_ensemble.py \
--test_image_name image_name \
--test_method five \
--output_name output.txt
```

## Citation
If you find this code is useful for your research, please cite:
```
@article{sun2023blind,
  title={Blind quality assessment for in-the-wild images via hierarchical feature fusion and iterative mixed database training},
  author={Sun, Wei and Min, Xiongkuo and Tu, Danyang and Ma, Siwei and Zhai, Guangtao},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2023},
  publisher={IEEE}
}
```
