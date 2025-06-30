import torch
import tool_model as tm
import tool_train as tt
import tool_test_and_draw as td

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
离散属性类别数
{'2022年分类单元土壤污染等级': 4, '2022年分类单元土层厚度': 4, '2022年分类单元土壤质地': 4, '2022年扩充分类单元土壤污染等级': 4, '2022年扩充分类单元土层厚度': 4, '2022年扩充分类单元土壤
质地': 4, '2022年坡度是否在2°以上': 2, '2022年坡度是否在6°以上': 2, '2022年坡度是否在15°以上': 2, '2022年坡度是否在25°以上': 2, '2022年主体功能区': 6, '2022年饮用水保护区等级': 4, '2022 年主体功能区2017版': 6}
(4,4,4,4,4,4,2,2,2,2,6,4,6)
"""
name = "yuexi_"
# name_list = [ "yuedong_", "yuexi_", "zhusanjiao_", "yuebei_"]

train_loader, test_loader, val_loader, weights = tt.creat_loader(
    batch_size=256, name=name
)
print(weights)

import sys


for k in range(5):
    model = tm.SWIN_T_base_Model(num_continuous=68 - 12).to(device)
    model.read_and_freeze_tabmodel_parameters(
        "pre_train_tab_contrast.pth"
    )
    output_file = "contrast_with_train_result" + str(k) + ".txt"
    # 打开输出文件，并将标准输出重定向到文件
    with open(output_file, "w") as f:
        sys.stdout = f
        for i in range(3):
            tt.train_model_contrast(
                model, train_loader, device, class_weights=weights.to(device), epochs=10
            )
            print("*******************" + name + "test集效果" + "*******************")
            all_labels, all_predictions, probe = td.test_model(
                model, test_loader, device
            )
            print("*******************" + name + "val集效果" + "*******************")
            all_labels, all_predictions, probe = td.test_model(
                model, val_loader, device
            )
            print("*******************" + name + "train集效果" + "*******************")
            all_labels, all_predictions, probe = td.test_model(
                model, train_loader, device
            )
            torch.save(model.state_dict(), "param_model_SAFEPP.pth")

    # 恢复标准输出
    sys.stdout = sys.__stdout__




