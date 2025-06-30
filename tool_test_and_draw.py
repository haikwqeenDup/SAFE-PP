import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import pandas as pd


def test_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向传播获取模型输出
            predictions = model(X_batch.float())
            # 使用 softmax 获取每个类别的概率
            softmax_probabilities = torch.softmax(predictions, dim=1)
            # 获取预测类别，取最大概率的索引
            predicted_labels = torch.argmax(softmax_probabilities, dim=1).cpu().numpy()

            # 记录所有预测和标签
            all_predictions.extend(predicted_labels)
            all_labels.extend(y_batch.cpu().numpy())
            all_probabilities.extend(softmax_probabilities.cpu().numpy())
    # 计算准确性
    accuracy = accuracy_score(all_labels, all_predictions)
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    # 打印分类报告
    report = classification_report(all_labels, all_predictions)

    if len(all_probabilities) > 0:
        # 对于二分类问题，取正类的概率
        auc_score = roc_auc_score(all_labels, [prob[1] for prob in all_probabilities])
    else:
        auc_score = 0.0
    # 打印结果
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print(f"AUC: {auc_score:.4f}")

    report = classification_report(all_labels, all_predictions, output_dict=True)
    for label, metrics in report.items():
        if label != "accuracy" and (label == "0.0" or label == "1.0"):
            print(
                f"Label: {label} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-score: {metrics['f1-score']:.4f}"
            )
    return all_labels, all_predictions, all_probabilities


def save_result(
    all_labels,
    all_predictions,
    all_probabilities,
    df_filename="figure_train_df.pkl",
    reslut_filename="",
):

    df = pd.read_pickle(df_filename)
    all_labels = [float(label) for label in all_labels]  # 确保是纯数字
    all_predictions = [float(pred) for pred in all_predictions]

    dim1 = [float(arr[0]) for arr in all_probabilities]
    dim2 = [float(arr[1]) for arr in all_probabilities]

    df["true_labels"] = all_labels
    df["predicted_labels"] = all_predictions
    df["p_0"] = dim1
    df["p_1"] = dim2

    temp_df: pd.DataFrame = df[
        ["geohash", "true_labels", "predicted_labels", "p_0", "p_1"]
    ]

    temp_df.to_csv(reslut_filename, index=False)
    print(f"DataFrame has been saved to excel.")


def draw_figure(
    all_labels,
    all_predictions,
    all_probabilities,
    df_figure_train_filename="figure_train_df.pkl",
    df_figure_test_filename="figure_test_df.pkl",
    filename="results.xlsx",
    for_train=False,
):
    df_figure_train = pd.read_pickle(df_figure_train_filename)
    df_figure_test = pd.read_pickle(df_figure_test_filename)

    if for_train:
        all_labels = [float(label) for label in all_labels]  # 确保是纯数字
        all_predictions = [float(pred) for pred in all_predictions]
        all_probabilities = [float(probability) for probability in all_probabilities]

        df_figure_train["true_labels"] = all_labels
        df_figure_train["predicted_labels"] = all_predictions
        df_figure_train["probability"] = all_probabilities

        temp_df: pd.DataFrame = df_figure_train[
            ["geohash", "true_labels", "predicted_labels", "probability"]
        ]
        temp_df.to_csv(filename, index=False)
        print(f"DataFrame has been saved to excel.")
        return

    # a = all_labels
    # b = all_predictions
    # merged = [
    #     (
    #         0
    #         if a[i] == 0 and b[i] == 0
    #         else (
    #             1
    #             if a[i] == 1 and b[i] == 0
    #             else (
    #                 2
    #                 if a[i] == 1 and b[i] == 1
    #                 else 3 if a[i] == 0 and b[i] == 1 else None
    #             )
    #         )
    #     )  # 默认值，处理未定义情况
    #     for i in range(len(a))
    # ]

    # df_figure_test["label"] = merged
    # plt.figure(figsize=(10, 8.5))
    # size = 50
    # # 绘制训练集
    # plt.scatter(
    #     df_figure_train.loc[df_figure_train["kzstmj"] != 1, "longitude"],
    #     df_figure_train.loc[df_figure_train["kzstmj"] != 1, "latitude"],
    #     color="red",  # 使用红色
    #     label="Train",
    #     alpha=0.5,
    #     s=size,
    # )

    # # 绘制 kzstmj 为 1 的训练集数据点
    # plt.scatter(
    #     df_figure_train.loc[df_figure_train["kzstmj"] == 1, "longitude"],
    #     df_figure_train.loc[df_figure_train["kzstmj"] == 1, "latitude"],
    #     color="black",  # 使用黑色
    #     label="Train (kzstmj=1)",
    #     alpha=1,
    #     s=size,
    # )

    # # 绘制测试集
    # plt.scatter(
    #     df_figure_test.loc[df_figure_test["label"] == 0, "longitude"],
    #     df_figure_test.loc[df_figure_test["label"] == 0, "latitude"],
    #     color="green",
    #     label="Test",
    #     alpha=0.25,
    #     s=size,
    # )

    # plt.scatter(
    #     df_figure_test.loc[df_figure_test["label"] == 1, "longitude"],
    #     df_figure_test.loc[df_figure_test["label"] == 1, "latitude"],
    #     color="purple",
    #     label="Test",
    #     alpha=1,
    #     s=size,
    #     marker="s",  # 使用方形标记
    # )
    # plt.scatter(
    #     df_figure_test.loc[df_figure_test["label"] == 2, "longitude"],
    #     df_figure_test.loc[df_figure_test["label"] == 2, "latitude"],
    #     color="black",
    #     label="Test",
    #     alpha=1,
    #     s=size,
    # )
    # plt.scatter(
    #     df_figure_test.loc[df_figure_test["label"] == 3, "longitude"],
    #     df_figure_test.loc[df_figure_test["label"] == 3, "latitude"],
    #     color="blue",
    #     label="Test",
    #     alpha=0.5,
    #     s=size,
    # )

    # # 添加标题和标签
    # plt.title("Latitude and Longitude of Train and Test Data")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")

    # # 添加图例
    # plt.legend()

    # # 显示网格
    # plt.grid()

    # # 展示图形
    # plt.tight_layout()
    # plt.show()

    all_labels = [float(label) for label in all_labels]  # 确保是纯数字
    all_predictions = [float(pred) for pred in all_predictions]
    all_probabilities = [float(probability) for probability in all_probabilities]
    df_figure_test["true_labels"] = all_labels
    df_figure_test["predicted_labels"] = all_predictions
    df_figure_test["probability"] = all_probabilities

    temp_df = df_figure_test[
        ["geohash", "true_labels", "predicted_labels", "probability"]
    ]
    temp_df.to_csv(filename, index=False)
    print(f"DataFrame has been saved to excel.")
