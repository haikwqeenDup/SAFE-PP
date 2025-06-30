import pandas as pd
import geohash
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def preprocess_csv(filename, result_name="name.pkl"):
    df = pd.read_csv(filename)
    df.rename(columns={"2022年垦造水田面积": "kzstmj"}, inplace=True)

    def geohash_to_latlng(geohash_t):
        lat, lng = geohash.decode(geohash_t)
        return lat, lng

    df["kzstmj"] = (df["kzstmj"] > 0).astype(int)
    print(df["kzstmj"].value_counts())
    df[["latitude", "longitude"]] = df["geohash"].apply(
        lambda x: pd.Series(geohash_to_latlng(x))
    )

    df.fillna(0, inplace=True)

    # X = df[["latitude", "longitude"]]
    # X_scaled = X
    # 使用 DBSCAN 进行聚类
    # dbscan = DBSCAN(eps=0.01, min_samples=1)  # 设置适当的 eps 和 min_samples 参数
    # df["cluster"] = dbscan.fit_predict(X_scaled)
    df["cluster"] = 0

    # 3. 可视化聚类结果
    # plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(
    #     df["longitude"],
    #     df["latitude"],
    #     c=df["cluster"],
    #     cmap="viridis",
    #     marker="o",
    #     alpha=0.6,
    # )

    # plt.title("DBSCAN Clustering of Latitude and Longitude")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.colorbar(scatter, label="Cluster Label")
    # plt.grid(True)
    # plt.show()
    df["lat_id"] = (
        df.groupby("cluster")["latitude"]
        .rank(method="dense", ascending=True)
        .astype(int)
        .apply(lambda x: x - 1)
    )
    df["lon_id"] = (
        df.groupby("cluster")["longitude"]
        .rank(method="dense", ascending=True)
        .astype(int)
        .apply(lambda x: x - 1)
    )
    df.to_pickle(result_name)

    print(df.head)
    missing_counts = df.isnull().sum()
    # 筛选缺失值个数大于零的列
    missing_columns = missing_counts[missing_counts > 0]
    # 打印结果
    print("Columns with missing values (count > 0):")
    print(missing_columns)
    print("Finished!")


import concurrent.futures


def main():
    # 定义文件路径和输出路径
    tasks = [
        (
            "datam/耕地智能分析2022年标签汇总_粤西_20241216版.csv",
            "preprocess_yuexi.pkl",
        ),
    ]

    # 使用 ProcessPoolExecutor 进行并行处理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(preprocess_csv, file_path, output_path)
            for file_path, output_path in tasks
        ]

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # 获取结果，抛出异常如果有的话
            except Exception as e:
                print(f"任务出现异常: {e}")


# kzstmj
# 0    1521719
# 1       8054

if __name__ == "__main__":
    main()
