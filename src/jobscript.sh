#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=8G
#SBATCH --job-name=output_TLINK_v2
module purge
module load scikit-learn

pip install --user -r requirements.txt
python train_model.py --train_path=matres_2_sentences/train_timebank.txt --epochs=50 --batch=16 --lr=0.01 --output_file=new_files/version_01_MLP_with_2sen --save_model=new_models/model_v_01_MLP_2sen --test_path=matres_2_sentences/test_platinum.txt --dev_path=matres_2_sentences/dev_aquaint.txt