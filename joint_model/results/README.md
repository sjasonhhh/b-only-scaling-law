# A+B联合模型

该目录是内部联合模型，不改动现有 `INTERFACE.md`。B侧预测以冻结的B-only模型为基线，A侧问题一的13域偏移指数模型只通过显式的配比修正器接入；A、B不按行拼接。

`joint_model.json` 是冻结参数；`evaluation_metrics.json` 分层记录A/B拟合、开发、隐藏反馈和压力结果。
A12-A15只用于规模偏移适配，B8/B9/B10仍单独报告，不进入主模型合格分。
