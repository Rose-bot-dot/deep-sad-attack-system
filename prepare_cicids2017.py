import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


RAW_DIR = r"data/CICIDS2017/raw"
PROCESSED_DIR = r"data/CICIDS2017/processed"
ATTACK_DATA_DIR = r"data/attack_data"
SAVED_MODELS_DIR = r"saved_models"


def load_all_csv(raw_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在目录 {raw_dir} 下没有找到 CSV 文件")

    dfs = []
    for file in csv_files:
        print(f"读取文件: {file}")
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 清理列名空格
    df.columns = [col.strip() for col in df.columns]

    # 清理字符串值两侧空格
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def build_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    if "Label" not in df.columns:
        raise ValueError("原始数据中没有 Label 列")

    df["label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    # 删除原始标签列
    drop_cols = ["Label"]

    # 常见不适合作为通用训练特征的标识性字段
    candidate_drop = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp"
    ]

    for col in candidate_drop:
        if col in df.columns:
            drop_cols.append(col)

    df = df.drop(columns=drop_cols, errors="ignore")

    # 只保留数值列 + label
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "label" not in numeric_cols:
        numeric_cols.append("label")

    df = df[numeric_cols].copy()

    # 去掉无穷值
    df = df.replace([np.inf, -np.inf], np.nan)

    # 删除全空列
    df = df.dropna(axis=1, how="all")

    # 保证 label 在最后
    feature_cols = [col for col in df.columns if col != "label"]
    df = df[feature_cols + ["label"]]

    return df, feature_cols


def split_dataset(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]

    # 先切出测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 再切出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_and_save(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ATTACK_DATA_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # 缺失值填补 + 标准化
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 生成 DataFrame
    train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df["label"] = y_train.values

    val_df = pd.DataFrame(X_val_scaled, columns=feature_cols)
    val_df["label"] = y_val.values

    test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df["label"] = y_test.values

    # 保存到系统使用目录
    train_df.to_csv(os.path.join(ATTACK_DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(ATTACK_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(ATTACK_DATA_DIR, "test.csv"), index=False)

    # 保存预处理器和特征列
    joblib.dump(
        {"imputer": imputer, "scaler": scaler},
        os.path.join(SAVED_MODELS_DIR, "preprocessor.joblib")
    )

    with open(os.path.join(SAVED_MODELS_DIR, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=4)

    print("数据预处理完成")
    print(f"训练集: {train_df.shape}")
    print(f"验证集: {val_df.shape}")
    print(f"测试集: {test_df.shape}")


def main():
    df = load_all_csv(RAW_DIR)
    print("合并后数据形状:", df.shape)

    df = clean_columns(df)
    df = build_binary_label(df)
    df, feature_cols = select_features(df)

    # 丢掉存在空标签的行
    df = df.dropna(subset=["label"])

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    preprocess_and_save(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_cols
    )


if __name__ == "__main__":
    main()