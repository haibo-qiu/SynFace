DATA_ROOT='data/datasets/CASIA'
CASIA_SAVE=${DATA_ROOT}'/CASIA-10k-100-01-MixID/'
CASIA_ALIGN=${DATA_ROOT}'/CASIA-10k-100-01-MixID-Crop/'

source /public/data1/software/anaconda3/etc/profile.d/conda.sh
conda activate tf

# images synthesis
python -u data/DiscoFaceGAN/syn_images.py --num_id 10000 \
                                          --samples_perid 100 \
                                          --save_path $CASIA_SAVE

#python -u data/DiscoFaceGAN/syn_factors.py  --save_path $CASIA_SAVE --factor 1

source /public/data1/software/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# images align and crop
python -u data/imgs_crop/face_align_crop.py -j 8 -source_root $CASIA_SAVE -dest_root $CASIA_ALIGN

