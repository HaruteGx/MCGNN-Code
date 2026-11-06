# SHAP 分析与结果

运行脚本可导出 SHAP 数值、汇总表及审计结果。论文图片见下方结果目录。

公开图片和表格统一入口为 `outputs/interpretability/shap`：

- `non_fingerprint_main`：论文使用的非指纹归因。温度和目标样本自身的指纹保持固定，对其余特征计算归因。
- `including_fingerprints_supplementary`：固定温度，指纹作为整体特征组参与归因的补充结果。
- `paper_composite`：论文中的消融、底物 IG 和非指纹全局 SHAP 合图，以及绘图源数据。

两个单独分析目录均包含：

- `all_inputs.png/svg`：全部归因输入组的平均绝对 SHAP 排名。
- `global_features.png/svg`：全局特征的平均绝对 SHAP 排名。
- `shap_distribution.png/svg`：有符号 SHAP 分布。
- `shap_values.csv`：每条测试记录各特征组的 SHAP、顺序抽样误差等。
- `feature_importance.csv`：平均绝对 SHAP 及排序。
- `sample_audit.csv`：基准预测、目标预测和归因加和等数值。
- `background_records.csv`：所用背景反应记录。

结果来自全部 251 条固定 Ir 测试记录，使用 8 条开发集背景记录和 16 对随机顺序/反向顺序。

`src/shap` 目录存放分析代码和说明。逐顺序对归因估计用于数值稳定性分析。

SHAP 值的单位为 kcal/mol，正值表示相对背景提高预测活性指标，负值表示降低；平均绝对值表示影响幅度。固定指纹模式在目标指纹保持固定的条件下计算其余特征的归因。

重新运行示例（从公开版根目录运行，使用新目录保存）：

```sh
python src/shap/run_shap.py --fingerprints fixed --output-dir outputs/interpretability/shap/reproduced_nonfingerprint
python src/shap/run_shap.py --output-dir outputs/interpretability/shap/reproduced_allfeatures
```

脚本默认结果目录为 `outputs/interpretability/shap`。已有结果目录不会被覆盖；重新运行时请使用上方示例指定一个新的空目录。
