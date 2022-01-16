#!/bin/bash

adult=(adult 10.0 "104 20 20" "20 20 104" 104_20_20_20_104)
compas=(compas 1.0 "14 20 20" "20 20 14" 14_20_20_20_14)
crime=(crime 10.0 "147 20 20" "20 20 147" 147_20_20_20_147)
german=(german 10.0 "59 20 20" "20 20 59" 59_20_20_20_59)
health=(health 1.0 "110 20 20" "20 20 110" 110_20_20_20_110)
lawschool=(lawschool 0.1 "37 20" "20 37" 37_20_37)

dataset="adult"
encoder_layers="104 20 20"
decoder_layers="20 20 104"
layers=104_20_20_20_104

# baseline
#python train_encoder.py --dataset ${dataset} --load \
#  --encoder-layers ${encoder_layers} \
#  --decoder-layers ${decoder_layers} \
#  --constraint "GeneralCategorical(0.01, 0.3, [])" \
#  --dl2-weight 0

# LCIFR data producer
python train_encoder.py --dataset ${dataset} --load \
  --encoder-layers ${encoder_layers} \
  --decoder-layers ${decoder_layers} \
  --constraint "GeneralCategorical(0.01, 0.3, [])" \
  --conditional_training true

# LCIFR data consumer
python train_classifier.py --dataset ${dataset} --load \
  --encoder-layers ${encoder_layers} \
  --decoder-layers ${decoder_layers} \
  --constraint "GeneralCategorical(0.01, 0.3, [])" \
  --load-epoch 99

# certify baseline
#python certify.py \
#  --models-dir "../../models/${dataset}/GeneralCategorical(0.01, 0.3, [])/${layers}/dl2_weight_0.0_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_0.0" \
#  --num-certify 1000 --complete --epoch 99 --load

# certify LCIFR
#python certify.py \
#  --models-dir "../../models/${dataset}/GeneralCategorical(0.01, 0.3, [])/${layers}/dl2_weight_${dl2_weight}_learning_rate_0.001_weight_decay_0.01_balanced_False_patience_5_quantiles_False_dec_weight_0.0" \
#  --num-certify 1000 --complete --epoch 99 --load --adversarial
