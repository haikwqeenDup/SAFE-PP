import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def split_grid(df: pd.DataFrame, grid_size):
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

    grid_size_lat = grid_size / 111.32
    grid_size_lon = grid_size / (
        111.32 * math.cos(math.radians(lat_max * 0.5 + lat_min * 0.5))
    )

    # 计算网格索引
    def get_grid_index(lat, lon, lat_min, lon_min):
        row = math.floor((lat - lat_min) / grid_size_lat)
        col = math.floor((lon - lon_min) / grid_size_lon)
        return (row, col)

    # 新增网格索引列
    df["grid_index"] = df.apply(
        lambda row: get_grid_index(row["latitude"], row["longitude"], lat_min, lon_min),
        axis=1,
    )

    # 使用 groupby 获取每个 grid_index 的标签集合
    grid_labels = df.groupby("grid_index")["kzstmj"].apply(set).to_dict()

    # 创建 grid_label 列，检查每个 grid_index 是否在 grid_labels 中
    df["grid_label"] = np.where(
        df["grid_index"].isin([k for k, v in grid_labels.items() if 1 in v]), 1, 0
    )
    print(f"划分得到的大网格数量：{(df['grid_index'].nunique())}")
    return df


def cat_coding(df, cat_col):
    for col in cat_col:
        df[col], unique = pd.factorize(df[col])
    return df

np.random.seed(42)
def split_train_test(df: pd.DataFrame, test_rate, random_state):
    df = df[df["grid_label"] != 0]  # df保留原有的信息，df_drop去掉无标签区域
    df.drop(columns=["grid_label"], inplace=True)

    # 获取所有唯一的grid_index值
    unique_grids = df['grid_index'].unique()
    fraction=0.2
    # 随机选择要保留的grid_index
    grids_to_keep = np.random.choice(
        unique_grids, 
        size=int(len(unique_grids) * fraction), 
        replace=False
    )
    
    # 保留选中组的数据
    df = df[df['grid_index'].isin(grids_to_keep)]
    # df = df.sample(frac=0.10, random_state=42)
    print(f"负采样后的大网格数量：{(df['grid_index'].nunique())}")
    grid_indices = df["grid_index"].unique()
    train_indices, test_indices = train_test_split(
        grid_indices, test_size=test_rate, random_state=random_state
    )

    train_df = df[df["grid_index"].isin(train_indices)]
    print(f"训练集大小：{len(train_df)}")
    test_df = df[df["grid_index"].isin(test_indices)]
    print(f"测试集大小：{len(test_df)}")
    return train_df, test_df


from multiprocessing import Pool, Manager


def process_subset(args):
    df, assist_df, subset_df, range_size, cat_col, name, is_train = args
    subset_array = np.zeros(
        (len(subset_df), len(df.columns), 2 * range_size + 1, 2 * range_size + 1)
    )

    for count, row in enumerate(subset_df.itertuples(), 1):
        if count % 1000 == 1:
            current_time = datetime.now()
            print(
                name
                + ("train" if is_train else "test")
                + f"构造进度：{count}/{len(subset_df)} | time {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        cluster = row.cluster
        lat_id = row.lat_id
        lon_id = row.lon_id
        latitude = row.latitude
        longitude = row.longitude
        for i in range(-range_size, range_size + 1):
            for j in range(-range_size, range_size + 1):
                try:
                    result = df.loc[(cluster, lat_id + i, lon_id + j)].values
                    temp_location = assist_df.loc[
                        (cluster, lat_id + i, lon_id + j)
                    ].values
                    origin_location = np.array([latitude, longitude])
                    if np.linalg.norm(temp_location - origin_location) < 0.02:
                        subset_array[count - 1, :, i + range_size, j + range_size] = (
                            result.astype(np.float64)
                        )
                except KeyError:
                    pass

    return subset_array


def create_range_tensor_mp(
    df,
    train_df,
    test_df,
    range_size,
    drop_col,
    cat_col,
    name="",
    num_processes=3,
    val_rate=0.5,
):
    val_df, testkkdf = train_test_split(test_df, test_size=val_rate, random_state=42)
    figure_val = val_df[["geohash", "kzstmj", "longitude", "latitude"]]
    figure_val.to_pickle("yeuxi_" + "figure_val_df.pkl")
    figure_test = testkkdf[["geohash", "kzstmj", "longitude", "latitude"]]
    figure_test.to_pickle("yeuxi_" + "figure_test_df.pkl")
    del val_df, figure_val, testkkdf, figure_test
    train_df = train_df[
        ["cluster", "lat_id", "lon_id", "kzstmj", "latitude", "longitude"]
    ]
    testa_df = test_df[
        ["cluster", "lat_id", "lon_id", "kzstmj", "latitude", "longitude"]
    ]
    assist_df = df[["cluster", "lat_id", "lon_id", "latitude", "longitude"]]
    val_df, test_df = train_test_split(testa_df, test_size=val_rate, random_state=42)
    del testa_df
    df = df.set_index(["cluster", "lat_id", "lon_id"])
    df = df.drop(columns=["grid_index", "latitude", "longitude", "kzstmj"] + drop_col)
    scaler = StandardScaler()
    df[[col for col in df.columns if col not in cat_col]] = scaler.fit_transform(
        df[[col for col in df.columns if col not in cat_col]]
    )
    assist_df = assist_df.set_index(["cluster", "lat_id", "lon_id"])

    train_splits = np.array_split(train_df, num_processes)
    test_splits = np.array_split(test_df, num_processes)
    val_splits = np.array_split(val_df, num_processes)
    with Pool(num_processes) as pool:
        train_results = pool.map(
            process_subset,
            [
                (df, assist_df, split, range_size, cat_col, name, True)
                for split in train_splits
            ],
        )
        X_train_tensor = torch.tensor(np.concatenate(train_results, axis=0))
        y_train_tensor = torch.tensor(train_df["kzstmj"].values)
        torch.save(X_train_tensor, name + "tensor_train_x.pt")
        torch.save(y_train_tensor, name + "tensor_train_y.pt")
        del train_results, X_train_tensor, y_train_tensor

        test_results = pool.map(
            process_subset,
            [
                (df, assist_df, split, range_size, cat_col, name, False)
                for split in test_splits
            ],
        )
        X_test_tensor = torch.tensor(np.concatenate(test_results, axis=0))
        y_test_tensor = torch.tensor(test_df["kzstmj"].values)
        torch.save(X_test_tensor, name + "tensor_test_x.pt")
        torch.save(y_test_tensor, name + "tensor_test_y.pt")
        del test_results, X_test_tensor, y_test_tensor

        val_results = pool.map(
            process_subset,
            [
                (df, assist_df, split, range_size, cat_col, name, False)
                for split in val_splits
            ],
        )
        X_val_tensor = torch.tensor(np.concatenate(val_results, axis=0))
        y_val_tensor = torch.tensor(val_df["kzstmj"].values)
        torch.save(X_val_tensor, name + "tensor_val_x.pt")
        torch.save(y_val_tensor, name + "tensor_val_y.pt")


def preprocess_pkl_to_train(
    filename,
    cat_col,
    drop_col,
    range_size=3,
    grid_size=3,
    test_rate=0.35,
    val_rate=0.5,
    random_state=123,
    pre_name="",
):
    df = pd.read_pickle(filename)
    df = split_grid(df, grid_size)
    df = cat_coding(df, cat_col)
    train_df, test_df = split_train_test(df, test_rate, random_state)

    figure_train = train_df[["geohash", "kzstmj", "longitude", "latitude"]]
    # figure_test = test_df[["geohash", "kzstmj", "longitude", "latitude"]]
    figure_train.to_pickle(pre_name + "figure_train_df.pkl")
    # figure_test.to_pickle(pre_name + "figure_test_df.pkl")
    df = df[cat_col + [col for col in df.columns if col not in cat_col]]
    create_range_tensor_mp(
        df,
        train_df,
        test_df,
        range_size,
        drop_col,
        cat_col,
        name=pre_name,
        num_processes=2,
        val_rate=val_rate,
    )
    print({col: df[col].nunique() for col in cat_col})


if __name__ == "__main__":
    preprocess_pkl_to_train(
        "preprocess_yuexi.pkl",
        cat_col=[
            "2022年分类单元土壤污染等级",
            "2022年分类单元土层厚度",
            "2022年分类单元土壤质地",
            "2022年扩充分类单元土壤污染等级",
            "2022年扩充分类单元土层厚度",
            "2022年扩充分类单元土壤质地",
            "2022年坡度是否在2°以上",
            "2022年坡度是否在6°以上",
            "2022年坡度是否在15°以上",
            # "2022年坡度是否在25°以上",
            "2022年主体功能区",
            "2022年饮用水保护区等级",
            "2022年主体功能区2017版",
        ],
        drop_col=[
            "geohash",
            "sjdm",
            "sjmc",
            "dq",
            "2022年耕地流出面积",
            "sjdm.1",
            "sjmc.1",
            "dq.1",
            "2022年单独选址项目",
            "2022年采矿及盐田用地面积",
            "2022年规划用地用海湿地面积",
            "2022年临时用地占用面积",
            "2022年规划用地用海未利用地面积",
            "2022年坡度是否在25°以上",
            "2022年交通运输用地面积",
            "2022年规划用地用海城镇建设用地面积",
            "2022年农业设施用地占用面积",
            "2022年风景名胜区及特殊用地面积",
            "2022年城市建设用地面积",
            "2022年建设用地审批",
            "2022年规划用地用海林地面积",
        ],
        grid_size=3,
        range_size=3,
        pre_name="yuexi_",
        test_rate=0.3,
        val_rate=0.5,
    )

    print("---------------yuexi finished---------------")
