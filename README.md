# B-only Scaling Law Model

这是问题二 B 组的可复现推理发布包，提供冻结后的
\(\widehat L_B(N,D,Q_{score})\) 模型参数和 CSV 预测脚本。

本仓库面向只需要使用模型预测的读者：不需要下载原始 B 组附件，也不需要重新拟合模型。
当前发布版只包含 B-only 模型；A 组配比变量不在本发布包中。

## 快速开始

```bash
git clone https://github.com/sjasonhhh/b-only-scaling-law-model.git
cd b-only-scaling-law-model

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m b_model.predict \
  --input examples/input.csv \
  --output predictions.csv
```

Windows PowerShell 激活虚拟环境的命令是：

```powershell
.venv\Scripts\Activate.ps1
```

## 输入格式

输入必须是 CSV，至少包含以下列：

| 列名 | 含义 | 单位/范围 |
|---|---|---|
| `N_params_B` | 模型参数量 | 十亿参数（B） |
| `D_tokens_B` | 训练数据量 | 十亿 token（B） |
| `Q_score` | 半合成质量分 | 可选；通常在 `[0,1]` |

如果提供 `Q_score`，脚本使用完整的
\(\widehat L_B(N,D,Q_{score})\)；如果不提供，则只使用基础的
\(\widehat L_B(N,D)\) 关系。

示例输入见 `examples/input.csv`。

## 输出格式

预测结果会保留输入列，并增加：

- `prediction`：Loss 点预测；
- `prediction_std`：冻结模型残差尺度；
- `prediction_low_90`、`prediction_high_90`：90%预测区间；
- `D_tokens_B_used`：实际送入模型的数据量；
- `D_tokens_B_imputed`：是否对非正 `D_tokens_B` 使用下限场景；
- `prediction_status`：`ok` 或 `D_floor_imputed`。

当 `D_tokens_B <= 0` 时，脚本不会伪造原始输入，而是使用 B1 中最小正数据量 `0.134` 作为下限场景，并在输出中明确标记。

## 模型参数

冻结参数文件位于：

```text
b_model/results/B_only_model.json
```

也可以显式指定其他参数文件：

```bash
python -m b_model.predict \
  --model path/to/B_only_model.json \
  --input path/to/input.csv \
  --output path/to/predictions.csv
```

## 开发者验证

```bash
python -m unittest discover -s b_model/tests -p 'test_*.py' -v
python -m py_compile b_model/pipeline.py b_model/predict.py
```

## 发布内容与边界

- `b_model/predict.py`：独立 CSV 预测入口；
- `b_model/pipeline.py`：模型特征、参数加载、评测和结果生成代码；
- `b_model/results/B_only_model.json`：冻结模型参数；
- `b_model/results/evaluation_metrics.json`：分层评测指标与门禁结果；
- `b_model/results/data_freeze_manifest.json`：数据哈希、角色、重叠检查和数据质量记录；
- `b_model/results/B组数据迭代报告.md`：中文评测报告。

原始实验附件没有随推理发布包上传；因此 `pipeline.py` 的完整重新拟合需要另行提供与冻结清单匹配的原始附件。按照本 README 进行预测只需要模型参数文件和 Python 依赖。

## 重要限制

- 该模型预测的是 B-only 关系，不代表已经完成含领域配比的最终关系；
- B2 存在 Cerebras 模型族的绝对 Loss 偏移；
- B8 只作为压力测试，校准数据和外推数据必须分开解释；
- B9 中缺失或非正训练数据量只按下限场景预测，不应解释为真实的零 token 训练。
