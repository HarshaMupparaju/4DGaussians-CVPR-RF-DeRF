import os
from pathlib  import Path
import pandas as pd

set_num = 4
scenes = ['Birthday' 'Painter' 'Remy' 'Theater' 'Train']
# scenes = ['Birthday']
# Read the training and testing set 
data_dirpath = Path('data/InterDigital')
train_test_sets_dirpath = data_dirpath / 'train_test_sets'

required_train_test_set_dirpath = train_test_sets_dirpath / f'set{set_num:02d}'

train_videos_filepath = required_train_test_set_dirpath / 'TrainVideosData.csv'
test_videos_filepath = required_train_test_set_dirpath / 'TestVideosData.csv'

train_videos_dataframe = pd.read_csv(train_videos_filepath)
test_videos_dataframe = pd.read_csv(test_videos_filepath)


for scene in scenes:
    train_video_nums = list(train_videos_dataframe[train_videos_dataframe['scene_name'] == scene]['pred_video_num'])
    test_video_nums = list(test_videos_dataframe[test_videos_dataframe['scene_name'] == scene]['pred_video_num'])

    train_video_nums_string = ','.join([str(num) for num in train_video_nums])
    test_video_nums_string = ','.join([str(num) for num in test_video_nums])

    # First run COLMAP 
    # os.system(f'bash colmap.sh {data_dirpath}/{scene} llff {train_video_nums_string} {test_video_nums_string}')

    #Point Cloud Downsample
    os.system(f'python scripts/downsample_point.py data/InterDigital/{scene}/colmap/dense/workspace/fused.ply data/InterDigital/{scene}/points3D_downsample2.ply')
    # output_path = f''
    #Training
    os.system(f'python train.py -s data/InterDigital/{scene} --port 6017 --expname "InterDigital/set{set_num}/{scene}" --configs arguments/InterDigital/{scene}.py --train_views {train_video_nums_string} --test_views {test_video_nums_string}')

    #Rendering
    os.system(f'python render.py --model_path "output/InterDigital/set{set_num}/{scene}"  --skip_train --configs arguments/InterDigital/{scene}.py --train_views {train_video_nums_string} --test_views {test_video_nums_string}')

    #Evaluation
    os.system(f'python metrics.py --model_path "output/InterDigital/set{set_num}/{scene}/"')

