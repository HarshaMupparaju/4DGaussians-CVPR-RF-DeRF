import os
from pathlib  import Path
import pandas as pd

set_num = 3
scenes = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'falme_steak', 'sear_steak']
# Read the training and testing set 
data_dirpath = Path('data/dynerf')
train_test_sets_dirpath = data_dirpath / 'train_test_sets'

required_train_test_set_dirpath = train_test_sets_dirpath / f'set{set_num:02d}'

train_videos_filepath = required_train_test_set_dirpath / 'TrainVideosData.csv'
test_videos_filepath = required_train_test_set_dirpath / 'TestVideosData.csv'

train_videos_dataframe = pd.read_csv(train_videos_filepath)
test_videos_dataframe = pd.read_csv(test_videos_filepath)

print()

for scene in scenes:
    train_video_nums = list(train_videos_dataframe[train_videos_dataframe['scene_name'] == scene]['pred_video_num'])
    test_video_nums = list(test_videos_dataframe[test_videos_dataframe['scene_name'] == scene]['pred_video_num'])

    train_video_nums_string = ','.join([str(num) for num in train_video_nums])
    test_video_nums_string = ','.join([str(num) for num in test_video_nums])

    # First run COLMAP 
    os.system(f'conda deactivate && conda activate colmap')
    os.system(f'bash colmap.sh {data_dirpath}/{scene} llff {train_video_nums_string} {test_video_nums_string}')
    os.system(f'conda deactivate && conda activate 4DGaussians')

    #Point Cloud Downsample
    os.system(f'python scripts/downsample_point.py data/dynerf/{scene}/colmap/dense/workspace/fused.ply data/dynerf/{scene}/points3D_downsample2.ply')

    #Training
    os.system(f'python train.py -s data/dynerf/{scene} --port 6017 --expname "dynerf/set{set_num}/{scene}" --configs arguments/dynerf/{scene}.py --train-views {train_video_nums_string} --test-views {test_video_nums_string}')

    #Rendering
    os.system(f'python render.py --model_path "output/dynerf/set{set_num}/{scene}"  --skip_train --configs arguments/dynerf/{scene}.py')

    #Evaluation
    os.system(f'python metrics.py --model_path "output/dynerf/set{set_num}/{scene}/"')

