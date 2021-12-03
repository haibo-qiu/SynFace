python -u test.py --cfg experiments/WebFace.yaml \
                   --gpus '0' \
                   --loss_type 'Arc' \
                   --dataset 'Syn' \
                   --dm 1  \
                   --num_id 10000 \
                   --samples_perid 50 \
                   --real_num_id 2000 \
                   --real_samples_perid 10 \
                   --batch_size 64 \
