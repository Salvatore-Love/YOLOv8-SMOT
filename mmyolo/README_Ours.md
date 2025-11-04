# If you have any questions, please feel free to contact me, my email address is 1127136201@qq.com


# Environmental preparation

- Step 1. Create and activate a conda environment.

```
conda create -n mmyolo python=3.8 -y
conda activate mmyolo
```

- Step 2. Install pytorch

**CUDA 11.3**
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

- Step 3. Install the dependencies

```
cd mmyolo
pip install -U openmim
mim install -r requirements/mminstall.txt
mim install -r requirements/albu.txt   # Install MMYOLO
mim install -v -e .
```
# Inferring directly using the checkpoints we provide (at mmyolo\workdirs\)
We have placed the results of the individual checkpoints inference below mmyolo \ workdirs \ output 
- Using small size can achieve AP50 0.706. 
It took 6.5 minutes (one RTX3090) to infer all images in mva2023_sod4bird_pub_test. 
```
CUDA_VISIBLE_DEVICES=1 python test_v8s.py --json-prefix dir/to/mva2023_sod4bird_pub_test/annotations/
```
Please specify the correct paths for --config, --checkpoint, and --json-prefix.

- Using medium size can achieve at least AP50 0.731 (achieved using the checkpoint of the 80th epoch ). 
It took 15 minutes (one RTX3090) to infer all images in mva2023_sod4bird_pub_test. 
```
CUDA_VISIBLE_DEVICES=1 python test_v8s.py --json-prefix dir/to/mva2023_sod4bird_pub_test/annotations/ 
```
Please specify the correct paths for --config, --checkpoint, and --json-prefix.

- The effect of using Large size is unknown as it has not been tested. 
It took 25 minutes (one RTX3090) to infer all images in mva2023_sod4bird_pub_test.
```
CUDA_VISIBLE_DEVICES=1 python test_v8s.py --json-prefix dir/to/mva2023_sod4bird_pub_test/annotations/
```
Please specify the correct paths for --config, --checkpoint, and --json-prefix.

To reproduce or exceed our results, follow the steps below: 
1. Firstly, install sahi to split the training set into a dataset consisting of 1280*1280 size images. 
```
pip install sahi
``` 
Use the following command line:
```
sahi coco slice --image_dir dir/to/images --dataset_json_path dataset.json --slice_size 1280 --overlap_ratio 0.25
``` 
Here dataset.json includes split_train_coco.json, split_val_coco.json, and merged_train.json. 
2. Use YOLOv8 as our model. Note that you should change the paths for the dataset and annotations accordingly. 
- Train YOLOv8 with **small size**
```
bash tools/dist_train.sh configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_bird.py 4 --amp
``` 
If you want the program to keep training: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash tools/dist_train.sh configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_bird.py 4 --amp >>./yolo8_s_4gpu_b8.log 2>&1 &
``` 
Note that if you train for 100 epochs, the best result may be in the 90th epoch. If you train for 200 epochs, the best result is likely to be between 80-120 epochs. Overtraining may lead to overfitting, and the last ten epochs may not be suitable for Mosaic and Mix-up in this competition.
The final checkpoint at the 110th epoch on the public test set achieved an AP50 of 70.6.
- Train YOLOv8 with **medium size**
```
bash tools/dist_train.sh configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_bird.py 4 --amp
``` 
If you want the program to keep training: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash tools/dist_train.sh configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_bird.py 4 --amp >>./yolo8_m_4gpu_b8.log 2>&1 &
```
In fact, during the competition, we did not complete the training and only used the checkpoint at the 80th epoch. It achieved an AP50 of 73.1, which is our result on the leaderboard. The 90th and 100th epochs may perform better. We have provided these checkpoints in our appendix, and inference is very fast. We hope you can give it a try.
- Train YOLOv8 with **large size**
```
bash tools/dist_train.sh configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_bird.py 4 --amp
``` 
If you want the program to keep training: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash tools/dist_train.sh configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_bird.py 4 --amp >>./yolo8_s_4gpu_b8.log 2>&1 &
``` 

From the above, it can be seen that increasing the size of the model can bring considerable benefits. After the end of the competition, we trained a large_size model with a batch_size of 6 and trained for 100 epochs using four GPUs due to limited computing resources. The checkpoints from the 80th to the 100th epoch are provided in the appendix. Testing accuracy using this model should be higher than the medium_size model used in our leaderboard submission.

If you want to exceed our results, you can increase the batch_size and the number of training epochs accordingly. For example, you can use a Large size model with batch_size=16, gpu_num=8, and epoch=200. We believe that this can achieve even better results than the first-ranked team on the leaderboard.

Remember, the final checkpoint is not necessarily the best, but it has a better generalization performance during the training session (such as the checkpoint at 100 epochs) 

3.  Testing 
The results obtained as described above can be inferred using the following code:
```
CUDA_VISIBLE_DEVICES=1 python test_v8s.py --json-prefix dir/to/mva2023_sod4bird_pub_test/annotations/
``` 
Please specify the correct paths for --config and --checkpoint.

4.  Get the final json file result 

The final generated json result file is generally generated under  MVA2023_Challenge/data/mva2023_sod4bird_train/annotations/