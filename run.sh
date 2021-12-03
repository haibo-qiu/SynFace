# if only 4 gpus available
python -u train.py --cfg experiments/WebFace.yaml \
                   --gpus '0,1,2,3' \
                   --loss_type 'Arc' \
                   --dataset 'Syn' \
                   --lr 0.2 \
                   --dm 1  \
                   --num_id 10000 \
                   --samples_perid 50 \
                   --real_num_id 2000 \
                   --real_samples_perid 10 \
                   --batch_size 64 \
                   --debug 0

# if 8 gpus available, using the same settings in our paper
#python -u train.py --cfg experiments/WebFace.yaml \
                   #--gpus '0,1,2,3,4,5,6,7' \
                   #--loss_type 'Arc' \
                   #--dataset 'Syn' \
                   #--lr 0.1 \
                   #--dm 1  \
                   #--num_id 10000 \
                   #--samples_perid 50 \
                   #--real_num_id 2000 \
                   #--real_samples_perid 10 \
                   #--batch_size 64 \
                   #--debug 0
