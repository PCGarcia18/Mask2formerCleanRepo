#!/bin/bash

# 1. Initialize Conda for bash scripts (adjust path if your conda is installed elsewhere)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mask2former

# 2. Define variables
experiment_name="mask2former_rios"
number_repeat=5

# 3. Loop N times
for i in $(seq 1 $number_repeat); do

    echo "========================================================="
    echo " Starting Run ${i} of ${number_repeat} for ${experiment_name}"
    echo "========================================================="

    # Execute the python script
    python train_net_rios_16bit_raw_small_new_sampler.py \
        --num-gpus 1 \
        --config-file  /home/pablo.canosa/ssd/code_tests/Mask2formerCleanRepo/Mask2Former/configs/rios/sem-seg-band-weight-AVG/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
        OUTPUT_DIR ./pruebas_swin_weight_initialization_RGB_AVG_AVG/rios_${i}_augmented_5ch_AVG 

    echo " Run ${i} completed!"
    echo ""

done

echo "All ${number_repeat} runs finished successfully."

# /home/pablo.canosa/ssd/code_tests/Mask2formerCleanRepo/Mask2Former/configs/rios/sem-seg-band-weight-reordering/swin/maskformer2_swin_tiny_bs16_50ep.yaml
# /home/pablo.canosa/ssd/code_tests/Mask2formerCleanRepo/Mask2Former/configs/rios/sem-seg-band-weight-AVG/swin/maskformer2_swin_tiny_bs16_50ep.yaml
# /home/pablo.canosa/ssd/code_tests/Mask2formerCleanRepo/Mask2Former/configs/rios/sem-seg-3dim/swin/maskformer2_swin_tiny_bs16_50ep.yaml