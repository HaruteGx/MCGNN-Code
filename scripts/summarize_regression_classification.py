import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.regression_classification import attach_classification, export_classification, BOUNDARIES, RULE_VERSION
from src.regression_plotting import interval_metrics, dump_json


def sources(root):
    yield root / 'outputs/ir_prediction/tables/ir_models_predictions.csv', None, None, None
    co = root / 'outputs/co_transfer/evaluations'
    for name, role in [('development_oof', 'development_oof'), ('holdout_test', 'holdout_test')]:
        yield co / name / 'predictions.csv', 'Co', 'Co_transfer_joint', role


def run(root=ROOT, update_existing=False, plots=True):
    output = root/'outputs/regression_classification'
    output.mkdir(parents=True, exist_ok=True)
    tables, provenance = [], []
    for path, dataset, model, role in sources(root):
        original = pd.read_csv(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        augmented = attach_classification(original, dataset)
        if 'status' not in augmented:


            augmented['status'] = 'OK'
        if update_existing:
            augmented.to_csv(path, index=False, encoding='utf-8-sig')

            check = pd.read_csv(path)
            for col in ('Predicted G (kcal/mol)', 'predicted_delta_g_kcal_mol'):
                if col in original:
                    np.testing.assert_allclose(check[col], original[col], rtol=1e-14, atol=1e-14)
        provenance.append({'path': path.relative_to(root).as_posix(), 'rows': len(original),
                           'before_sha256': before, 'after_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        frame = augmented.rename(columns={'sample_name': 'Sample Name', 'Name': 'Sample Name',
            'true_delta_g_kcal_mol': 'True G (kcal/mol)', 'predicted_delta_g_kcal_mol': 'Predicted G (kcal/mol)',
            'fold': 'Fold', 'method': 'model'}).copy()
        if 'True G (kcal/mol)' not in frame:
            frame['True G (kcal/mol)'] = frame['G (kcal/mol)']
        if dataset:
            frame['dataset'], frame['model'], frame['evaluation'] = dataset, model, role
        frame.loc[frame.model.eq('IrOnly_ZeroShot_RF'), 'evaluation'] = 'zero_shot'
        frame['run'] = path.parent.name
        frame['source'] = path.relative_to(root).as_posix()
        columns = ['dataset', 'model', 'evaluation', 'run', 'source', 'Sample Name', 'Fold',
                   'status', 'reason',
                   'Temperature (K)', 'Predicted Delta H (kcal/mol)', 'Predicted Delta S (cal/mol/K)',
                   'True G (kcal/mol)', 'Predicted G (kcal/mol)', 'True Class', 'Predicted Class',
                   'Class Correct', 'Class Lower Boundary (kcal/mol)', 'Class Upper Boundary (kcal/mol)',
                   'Classification Rule']
        tables.append(frame[[c for c in columns if c in frame]])
    all_predictions = pd.concat(tables, ignore_index=True)


    invalid = all_predictions['True Class'].isna() | all_predictions['Predicted Class'].isna()
    if invalid.any():
        raise ValueError(
            f'{int(invalid.sum())} incomplete or failed predictions in the saved evaluation '
            'tables; cannot report a complete test/OOF comparison. Inspect source status/reason.'
        )
    all_predictions = export_classification(all_predictions, output,
        group_columns=('model', 'evaluation', 'run'), plots=plots,
        role='Saved regressions: Ir 23/25; Co 24/26 (test, OOF and zero-shot kept separate)')
    metrics, fold_metrics = [], []
    group_keys = ['dataset', 'model', 'evaluation', 'run']
    for key, frame in all_predictions.groupby(group_keys, dropna=False, sort=False):
        if frame['Sample Name'].duplicated().any():
            raise ValueError(f'Duplicate sample predictions in {key}')
        m = interval_metrics(frame['True G (kcal/mol)'].to_numpy(float),
                             frame['Predicted G (kcal/mol)'].to_numpy(float), BOUNDARIES[key[0]])
        metrics.append({**dict(zip(group_keys, key)), **{k: m[k] for k in
            ('n', 'correct', 'accuracy', 'balanced_accuracy', 'macro_f1', 'g_mae', 'g_rmse', 'g_r2')},
            **{f'class_{i}_recall': m['recalls'][i] for i in range(3)}})
        if 'oof' in key[2]:
            for fold, part in frame.groupby('Fold'):
                fm = interval_metrics(part['True G (kcal/mol)'].to_numpy(float),
                                      part['Predicted G (kcal/mol)'].to_numpy(float), BOUNDARIES[key[0]])
                fold_metrics.append({**dict(zip(group_keys, key)), 'fold': int(fold),
                                     **{k: fm[k] for k in ('n', 'correct', 'accuracy', 'balanced_accuracy', 'macro_f1', 'g_mae', 'g_rmse', 'g_r2')}})
    summary = pd.DataFrame(metrics)
    summary.to_csv(output/'summary_metrics.csv', index=False, encoding='utf-8-sig')
    all_predictions.to_csv(output/'all_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_metrics).to_csv(output/'fold_metrics.csv', index=False, encoding='utf-8-sig')
    dump_json(output/'provenance.json', {'rule': RULE_VERSION, 'boundaries': BOUNDARIES,
              'sources': provenance, 'note': 'No refitting; labels are derived after continuous regression.'})

    for dataset, model, role in [('Ir', 'Regress', 'existing_test'), ('Co', 'Regress transfer', 'grouped_oof')]:
        part = all_predictions[all_predictions.dataset.eq(dataset) & all_predictions.model.eq(model)
                               & all_predictions.evaluation.eq(role)]
        export_classification(part, output, dataset, stem=f'{dataset.lower()}_main',
                              role=f'{dataset} {model}: {role}', plots=plots)
    lines = ['# 固定阈值回归与分类结果', '',
             '框架：图模型预测ΔH、ΔS，再计算ΔG；RDKit基线直接预测ΔG；分类仅在回归预测之后进行。',
             'Ir阈值23/25，Co阈值24/26 kcal/mol。边界等号归入较高类别。没有分类模型训练。', '',
             '| 数据 | 模型 | 评估 | 正确/总数 | 准确率 | 平衡准确率 | ΔG MAE |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.dataset} | {row.model} | {row.evaluation} | {row.correct}/{row.n} | '
                     f'{row.accuracy:.2%} | {row.balanced_accuracy:.2%} | {row.g_mae:.4f} |')
    lines += ['', 'Ir固定测试与开发OOF分开。',
              'Co为139条分组OOF，零样本模型单独标注。CoOnly_RF与RDKit RandomForest为重复对照。',
              '此前阈值已基于既有预测被查看；本结果不是新的外部验证。',
              'summary_metrics.csv包含回归和分类指标；all_predictions.csv包含逐样本G、H/S（如有）和分类结果。']
    (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary.to_string(index=False))
    print('Saved:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-existing', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    run(update_existing=args.update_existing, plots=not args.no_plots)
