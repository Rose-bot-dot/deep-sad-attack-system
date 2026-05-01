import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = 'data/attack_data'
OUTPUT_DIR = 'data/attack_data_demo'

TRAIN_TARGET = 8000
VAL_TARGET = 2000
TEST_TARGET = 2000


def stratified_sample(df, target_size, label_col='label', random_state=42):
    if len(df) <= target_size:
        return df.copy()

    # 分层抽样
    sampled_df, _ = train_test_split(
        df,
        train_size=target_size,
        stratify=df[label_col],
        random_state=random_state
    )
    return sampled_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(INPUT_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

    train_demo = stratified_sample(train_df, TRAIN_TARGET)
    val_demo = stratified_sample(val_df, VAL_TARGET)
    test_demo = stratified_sample(test_df, TEST_TARGET)

    train_demo.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_demo.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_demo.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

    print('演示数据集已生成：')
    print('train:', train_demo.shape)
    print('val:', val_demo.shape)
    print('test:', test_demo.shape)


if __name__ == '__main__':
    main()