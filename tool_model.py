import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
import torch.nn.functional as F
import swin_transformer_v2 as sw


class CNN(nn.Module):
    def __init__(self, input_channels, output_dim, range_size):
        super(CNN, self).__init__()
        # 定义 CNN 模块的卷积层和池化层
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=0)
        self.fc = nn.Linear(
            8 * (range_size * 2 - 1) * (range_size * 2 - 1), output_dim
        )  # 假设特征图尺寸为16x16，请根据实际情况调整

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)  # 展平为全连接层输入
        x = self.fc(x)
        return x


class CNNbase_Model(nn.Module):
    def __init__(
        self,
        categories=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
        num_continuous=67,
        range_size=3,
    ):
        super(CNNbase_Model, self).__init__()
        # 初始化各个模块
        self.tab_transformer = TabTransformer(
            categories=categories,  # 分类特征类别数目
            num_continuous=num_continuous,  # 连续特征数目
            dim=12,
            dim_out=10,
            depth=3,
            heads=2,
            attn_dropout=0.2,
            ff_dropout=0.2,
            mlp_hidden_mults=(3, 2),
            mlp_act=nn.ReLU(),  # 激活函数
        )  # 假设 tab_model 已定义
        self.cnn = CNN(10, 10, range_size=range_size)
        # 定义全连接层
        self.fc_hidden = nn.Linear(20, 10)  # 第一个线性层
        self.fc_final = nn.Linear(10, 2)  # 第二个线性层
        self.relu = nn.ReLU()  # 激活函数
        self.cat_len = len(categories)
        self.range_size = range_size

    def contrast_f(self, img_data):
        pixel_vectors = img_data
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        return tab_features

    def forward(self, img_data):
        pixel_vectors = img_data.reshape(
            img_data.shape[0], img_data.shape[1], -1
        ).permute(0, 2, 1)
        pixel_vectors = pixel_vectors.reshape(
            img_data.shape[0] * img_data.shape[2] * img_data.shape[3], img_data.shape[1]
        )
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征
        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)

        output_tensor = tab_features.reshape(
            -1, img_data.shape[2], img_data.shape[3], 10
        )
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        cnn_features = self.cnn(output_tensor)

        # 提取特征以进行拼接
        tab_features = output_tensor[
            :, :, self.range_size, self.range_size
        ]  # 只提取中心特征

        # 拼接特征
        concatenated_features = torch.cat((tab_features, cnn_features), dim=1)
        # 全连接层前向传播
        output = self.fc_hidden(concatenated_features)
        output = self.relu(output)  # 激活函数
        output = self.fc_final(output)  # 最后一层
        return output


class SWIN_T_base_Model(nn.Module):
    def __init__(
        self,
        categories=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
        num_continuous=67,
    ):
        super(SWIN_T_base_Model, self).__init__()
        # 初始化各个模块
        self.tab_transformer = TabTransformer(
            categories=categories,  # 分类特征类别数目
            num_continuous=num_continuous,  # 连续特征数目
            dim=4,  # 嵌入维度
            dim_out=16,  # 输出维度
            depth=3,  # 中等网络深度
            heads=2,  # 多头注意力数目
            attn_dropout=0.25,  # 注意力 Dropout
            ff_dropout=0.25,  # Feed-Forward Dropout
            mlp_hidden_mults=(3, 2, 1),  # MLP 隐藏层大小
            mlp_act=nn.ReLU(),  # 使用 ReLU 激活函数
        )
        self.swin = sw.SwinTransformerV2(
            img_size=8,
            patch_size=2,
            in_chans=16,  # 修改为图像的实际通道数
            num_classes=16,  # 根据分类任务保留
            embed_dim=24,  # 降低嵌入维度以适配小输入
            depths=[2, 4],  # 增加网络深度
            num_heads=[2, 4],  # 增加注意力头数
            window_size=4,  # 使用更大的窗口以覆盖更多上下文信息
            mlp_ratio=3.0,  # 调整 MLP 比例
            qkv_bias=True,
            drop_rate=0.05,  # 减少正则化强度
            attn_drop_rate=0.05,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[4, 4],  # 与新的窗口大小匹配
        )

        # 定义全连接层
        self.fc_hidden = nn.Linear(32, 8)  # 第一个线性层
        self.fc_final = nn.Linear(8, 2)  # 第二个线性层
        self.relu = nn.ReLU()  # 激活函数
        self.cat_len = len(categories)
        fc_layers = []
        in_features = 16
        for _ in range(2):
            fc_layers.append(nn.Linear(in_features, 32))
            fc_layers.append(nn.ReLU())
            in_features = 32
        # 最后一层输出与输入形状一致
        fc_layers.append(nn.Linear(32, 16))
        self.fc = nn.Sequential(*fc_layers)

    def read_and_freeze_tabmodel_parameters(
        self, parafile="pretrain_para/pre_train_tab.pth"
    ):
        # 加载状态字典
        state_dict = torch.load(parafile)

        # 加载到模型中
        self.tab_transformer.load_state_dict(state_dict)

        # # 冻结参数
        # for param in self.tab_transformer.parameters():
        #     param.requires_grad = False

        # print(f"Parameters of Tabtransformer are now frozen.")

    def contrast_f(self, img_data):
        pixel_vectors = img_data
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        return tab_features

    def for_tsne_f(self, img_data):
        pixel_vectors = img_data.reshape(
            img_data.shape[0], img_data.shape[1], -1
        ).permute(0, 2, 1)
        pixel_vectors = pixel_vectors.reshape(
            img_data.shape[0] * img_data.shape[2] * img_data.shape[3], img_data.shape[1]
        )
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        tab_features = self.fc(tab_features)
        # 重塑特征以适应输入
        output_tensor = tab_features.reshape(
            -1, img_data.shape[2], img_data.shape[3], tab_features.shape[1]
        )
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        kk = F.pad(output_tensor, (1, 0, 0, 1), mode="constant", value=0)
        swin_features = self.swin(kk)

        # 提取特征以进行拼接
        tab_features = output_tensor[:, :, 3, 3]
        # 拼接特征
        concatenated_features = torch.cat((tab_features, swin_features), dim=1)
        # 全连接层前向传播
        output = self.fc_hidden(concatenated_features)
        return output

    def forward(self, img_data):
        pixel_vectors = img_data.reshape(
            img_data.shape[0], img_data.shape[1], -1
        ).permute(0, 2, 1)
        pixel_vectors = pixel_vectors.reshape(
            img_data.shape[0] * img_data.shape[2] * img_data.shape[3], img_data.shape[1]
        )
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        tab_features = self.fc(tab_features)
        # 重塑特征以适应 CNN 输入
        output_tensor = tab_features.reshape(
            -1, img_data.shape[2], img_data.shape[3], tab_features.shape[1]
        )  # 变为 (N, 5, 5, 10)
        output_tensor = output_tensor.permute(0, 3, 1, 2)

        kk = F.pad(output_tensor, (1, 0, 0, 1), mode="constant", value=0)
        swin_features = self.swin(kk)

        # 提取特征以进行拼接
        tab_features = output_tensor[:, :, 3, 3]

        # 拼接特征
        concatenated_features = torch.cat((tab_features, swin_features), dim=1)
        # 全连接层前向传播
        output = self.fc_hidden(concatenated_features)
        output = self.relu(output)  # 激活函数
        output = self.fc_final(output)  # 最后一层

        return output


class MLPClassifier(nn.Module):
    def __init__(
        self,
        num_continuous,
        cat_cardinalities=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
        embedding_dim=4,
    ):
        """
        Args:
            num_continuous (int): 连续属性的数量
            cat_cardinalities (list[int]): 每个类别属性的类别数量
            embedding_dim (int): 每个类别属性的嵌入维度
            hidden_dim (int): 隐藏层的维度大小
            output_dim (int): 输出类别数量
        """
        super(MLPClassifier, self).__init__()
        self.cat_len = len(cat_cardinalities)  # 类别属性的数量
        # 类别属性的嵌入层
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_classes, embedding_dim)
                for num_classes in cat_cardinalities
            ]
        )
        self.embedding_dim_total = embedding_dim * len(cat_cardinalities)  # 嵌入总维度

        # 第一层全连接层：输入是连续属性 + 类别属性嵌入
        self.fc1 = nn.Linear(num_continuous + self.embedding_dim_total,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4  = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 2)  # 最后一层输出为2类
        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def for_tsne_f(self, img_data):
        pixel_vectors = img_data[:, :, 3, 3]
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征
        categorical_inputs = categ_input
        continuous_inputs = cont_input
        # 处理类别属性（嵌入并拼接）
        embedded = [
            embedding(categorical_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        embedded = torch.cat(embedded, dim=1)

        # 拼接连续属性和嵌入后的类别属性
        x = torch.cat([continuous_inputs, embedded], dim=1)

        # MLP 前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

    def forward(self, img_data):
        pixel_vectors = img_data[:, :, 3, 3]
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征
        categorical_inputs = categ_input
        continuous_inputs = cont_input
        # 处理类别属性（嵌入并拼接）
        embedded = [
            embedding(categorical_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        embedded = torch.cat(embedded, dim=1)

        # 拼接连续属性和嵌入后的类别属性
        x = torch.cat([continuous_inputs, embedded], dim=1)

        # MLP 前向传播
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


class MLP_RP_TAB(nn.Module):
    def __init__(
        self,
        categories=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
        num_continuous=67,
    ):
        super(MLP_RP_TAB, self).__init__()
        # 初始化各个模块
        self.tab_transformer = MLPClassifier(
            num_continuous=num_continuous, cat_cardinalities=categories
        )

        self.swin = sw.SwinTransformerV2(
            img_size=8,
            patch_size=2,
            in_chans=16,  # 修改为图像的实际通道数
            num_classes=16,  # 根据分类任务保留
            embed_dim=24,  # 降低嵌入维度以适配小输入
            depths=[2, 4],  # 增加网络深度
            num_heads=[2, 4],  # 增加注意力头数
            window_size=4,  # 使用更大的窗口以覆盖更多上下文信息
            mlp_ratio=3.0,  # 调整 MLP 比例
            qkv_bias=True,
            drop_rate=0.05,  # 减少正则化强度
            attn_drop_rate=0.05,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[4, 4],  # 与新的窗口大小匹配
        )

        # 定义全连接层
        self.fc_hidden = nn.Linear(32, 8)  # 第一个线性层
        self.fc_final = nn.Linear(8, 2)  # 第二个线性层
        self.relu = nn.ReLU()  # 激活函数
        self.cat_len = len(categories)
        fc_layers = []
        in_features = 16
        for _ in range(2):
            fc_layers.append(nn.Linear(in_features, 32))
            fc_layers.append(nn.ReLU())
            in_features = 32
        # 最后一层输出与输入形状一致
        fc_layers.append(nn.Linear(32, 16))
        self.fc = nn.Sequential(*fc_layers)

    def read_and_freeze_tabmodel_parameters(
        self, parafile="pretrain_para/pre_train_tab.pth"
    ):
        # 加载状态字典
        state_dict = torch.load(parafile)

        # 加载到模型中
        self.tab_transformer.load_state_dict(state_dict)

        # # 冻结参数
        # for param in self.tab_transformer.parameters():
        #     param.requires_grad = False

        # print(f"Parameters of Tabtransformer are now frozen.")

    def contrast_f(self, img_data):
        pixel_vectors = img_data
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        return tab_features

    def forward(self, img_data):
        pixel_vectors = img_data.reshape(
            img_data.shape[0], img_data.shape[1], -1
        ).permute(0, 2, 1)
        pixel_vectors = pixel_vectors.reshape(
            img_data.shape[0] * img_data.shape[2] * img_data.shape[3], img_data.shape[1]
        )
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        tab_features = self.fc(tab_features)
        output_tensor = tab_features.reshape(
            -1, img_data.shape[2], img_data.shape[3], tab_features.shape[1]
        )  # 变为 (N, 5, 5, 10)
        output_tensor = output_tensor.permute(0, 3, 1, 2)

        kk = F.pad(output_tensor, (1, 0, 0, 1), mode="constant", value=0)
        swin_features = self.swin(kk)

        # 提取特征以进行拼接
        tab_features = output_tensor[:, :, 3, 3]
        # 拼接特征
        concatenated_features = torch.cat((tab_features, swin_features), dim=1)
        # 全连接层前向传播
        output = self.fc_hidden(concatenated_features)
        output = self.relu(output)  # 激活函数
        output = self.fc_final(output)  # 最后一层

        return output


class Single_Tab(nn.Module):
    def __init__(
        self,
        categories=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
        num_continuous=67,
    ):
        super(Single_Tab, self).__init__()
        # 初始化各个模块
        self.tab_transformer = TabTransformer(
            categories=categories,  # 分类特征类别数目
            num_continuous=num_continuous,  # 连续特征数目
            dim=4,  # 嵌入维度
            dim_out=16,  # 输出维度
            depth=3,  # 中等网络深度
            heads=2,  # 多头注意力数目
            attn_dropout=0.25,  # 注意力 Dropout
            ff_dropout=0.25,  # Feed-Forward Dropout
            mlp_hidden_mults=(3, 2),  # MLP 隐藏层大小
            mlp_act=nn.ReLU(),  # 使用 ReLU 激活函数
        )
        fc_layers = []
        in_features = 16
        for _ in range(2):
            fc_layers.append(nn.Linear(in_features, 32))
            fc_layers.append(nn.ReLU())
            in_features = 32
        # 最后一层输出与输入形状一致
        
        self.fc = nn.Sequential(*fc_layers)
        self.last_fc = nn.Linear(32, 8)  # 最后一层输出
        self.relu = nn.ReLU()
        self.last_out = nn.Linear(8, 2)
        self.cat_len = len(categories)

    def for_tsne_f(self, img_data):
        pixel_vectors = img_data[:, :, 3, 3]
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        tab_features = self.fc(tab_features)
        # 全连接层前向传播
        output = self.last_fc(tab_features)
        return output

    def forward(self, img_data):
        pixel_vectors = img_data[:, :, 3, 3]
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        tab_features = self.fc(tab_features)
        # 全连接层前向传播
        output = self.last_fc(tab_features)
        output = self.relu(output)
        output = self.last_out(output)  # 最后一层
        return output


class CNN8x8Binary(nn.Module):
    def __init__(self):
        super(CNN8x8Binary, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_RP_SWIN(nn.Module):
    def __init__(
        self,
        categories=(20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20),
        num_continuous=67,
    ):
        super(CNN_RP_SWIN, self).__init__()
        # 初始化各个模块
        self.tab_transformer = TabTransformer(
            categories=categories,  # 分类特征类别数目
            num_continuous=num_continuous,  # 连续特征数目
            dim=4,  # 嵌入维度
            dim_out=16,  # 输出维度
            depth=3,  # 中等网络深度
            heads=2,  # 多头注意力数目
            attn_dropout=0.25,  # 注意力 Dropout
            ff_dropout=0.25,  # Feed-Forward Dropout
            mlp_hidden_mults=(3, 2),  # MLP 隐藏层大小
            mlp_act=nn.ReLU(),  # 使用 ReLU 激活函数
        )
        fc_layers = []
        in_features = 16
        for _ in range(2):
            fc_layers.append(nn.Linear(in_features, 32))
            fc_layers.append(nn.ReLU())
            in_features = 32
        # 最后一层输出与输入形状一致
        fc_layers.append(nn.Linear(32, 16))
        self.fc = nn.Sequential(*fc_layers)
        self.cnn = CNN8x8Binary()
        # 定义全连接层
        self.fc_hidden = nn.Linear(32, 8)  # 第一个线性层
        self.fc_final = nn.Linear(8, 2)  # 第二个线性层
        self.relu = nn.ReLU()  # 激活函数
        self.cat_len = len(categories)

    def contrast_f(self, img_data):
        pixel_vectors = img_data
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        return tab_features

    def forward(self, img_data):
        pixel_vectors = img_data.reshape(
            img_data.shape[0], img_data.shape[1], -1
        ).permute(0, 2, 1)
        pixel_vectors = pixel_vectors.reshape(
            img_data.shape[0] * img_data.shape[2] * img_data.shape[3], img_data.shape[1]
        )
        categ_input = pixel_vectors[:, : self.cat_len].long()  # 分类特征
        cont_input = pixel_vectors[:, self.cat_len :].float()  # 连续特征

        # TabTransformer 前向传播
        tab_features = self.tab_transformer(categ_input, cont_input)
        tab_features = self.fc(tab_features)
        output_tensor = tab_features.reshape(
            -1, img_data.shape[2], img_data.shape[3], tab_features.shape[1]
        )
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        kk = F.pad(output_tensor, (1, 0, 0, 1), mode="constant", value=0)
        swin_features = self.cnn(kk)

        # 提取特征以进行拼接
        tab_features = output_tensor[:, :, 3, 3]

        # 拼接特征
        concatenated_features = torch.cat((tab_features, swin_features), dim=1)
        # 全连接层前向传播
        output = self.fc_hidden(concatenated_features)
        output = self.relu(output)  # 激活函数
        output = self.fc_final(output)  # 最后一层
        return output

    def read_and_freeze_tabmodel_parameters(
        self, parafile="pretrain_para/pre_train_tab.pth"
    ):
        # 加载状态字典
        state_dict = torch.load(parafile)

        # 加载到模型中
        self.tab_transformer.load_state_dict(state_dict)
