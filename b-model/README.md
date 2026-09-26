# 联合广义标度律模型

## 模型

模型预测含模型规模、训练数据量、质量分和 17 维领域配比的 Loss：

```text
L_hat(N, D, Q, p_1, ..., p_17)
```

其中模型参数已冻结在 `model.json`，预测脚本为 `predict.py`。

## 快速开始

在仓库根目录执行：

```bash
python3 b-model/predict.py \
  --input b-model/examples/input.csv \
  --output predictions.csv
```

不需要创建虚拟环境，也不需要安装第三方 Python 包。

## 输入

输入必须是 CSV，并包含以下列：

| 列名 | 含义 | 要求 |
|---|---|---|
| `N_params_B` | 模型参数量 | 大于 0，单位为十亿参数（B） |
| `D_tokens_B` | 训练数据量 | 大于 0，单位为十亿 token（B） |
| `Q_score` | 质量分 | 范围为 `[0, 1]` |
| `p_1` 至 `p_17` | 17 个领域配比 | 非负，合计大于 0 |

脚本会将 `p_1` 至 `p_17` 自动归一化为配比向量。缺失字段、非数值、非有限值和非法范围会直接报错，不会自动补值。

示例输入见 `examples/input.csv`。

## 输出

输出 CSV 保留全部输入列，并新增一列：

| 列名 | 含义 |
|---|---|
| `prediction` | 联合模型给出的 Loss 点预测 |

## 文件

- `predict.py`：标准库预测脚本；
- `model.json`：冻结的联合模型参数；
- `examples/input.csv`：可直接运行的输入示例。
