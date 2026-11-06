from pathlib import Path

import numpy as np
import pandas as pd

BOUNDARIES = {'Ir': (23.0, 25.0), 'Co': (24.0, 26.0)}
RULE_VERSION = 'regression_intervals_ir23_25_co24_26_v1'
PREDICTED_G = 'Predicted G (kcal/mol)'
TRUE_G = 'True G (kcal/mol)'


def _column(frame, options, required=False):
    name = next((c for c in options if c in frame), None)
    if required and name is None:
        raise ValueError(f'Missing regression column; expected one of {options}')
    return name


def attach_classification(frame, dataset=None, default_catalyst=None):


    result = frame.copy()
    predicted_column = _column(result, (PREDICTED_G, 'predicted_delta_g_kcal_mol'), True)
    truth_column = _column(result, (TRUE_G, 'true_delta_g_kcal_mol', 'G (kcal/mol)'))
    if dataset is not None:
        metals = pd.Series(dataset, index=result.index, dtype='string')
    else:
        metal_column = _column(result, ('dataset', 'Cat', 'Classification Dataset'))
        metals = (result[metal_column].astype('string') if metal_column
                  else pd.Series(pd.NA, index=result.index, dtype='string'))
        metals = metals.str.strip().replace('', pd.NA).fillna(default_catalyst or pd.NA)
    metals = metals.str.strip().str.lower().map({'ir': 'Ir', 'co': 'Co'})
    if metals.isna().any():
        raise ValueError('Classification requires an explicit Ir/Co dataset or valid Cat per row.')
    predicted = pd.to_numeric(result[predicted_column], errors='coerce').to_numpy(float)
    truth = (pd.to_numeric(result[truth_column], errors='coerce').to_numpy(float)
             if truth_column else np.full(len(result), np.nan))
    valid = np.isfinite(predicted)
    if 'status' in result:
        valid &= result.status.eq('OK').to_numpy()
    true_labels = np.full(len(result), np.nan)
    pred_labels = np.full(len(result), np.nan)
    lower, upper = np.zeros(len(result)), np.zeros(len(result))
    for metal, cuts in BOUNDARIES.items():
        mask = metals.eq(metal).to_numpy()
        lower[mask], upper[mask] = cuts
        known = mask & np.isfinite(truth)
        successful = mask & valid
        true_labels[known] = np.searchsorted(cuts, truth[known], side='right')
        pred_labels[successful] = np.searchsorted(cuts, predicted[successful], side='right')
    result['Classification Dataset'] = metals.to_numpy()
    result['Class Lower Boundary (kcal/mol)'] = lower
    result['Class Upper Boundary (kcal/mol)'] = upper
    result['True Class'] = pd.array(true_labels, dtype='Int64')
    result['Predicted Class'] = pd.array(pred_labels, dtype='Int64')
    paired = np.isfinite(true_labels) & np.isfinite(pred_labels)
    correct = pd.array([pd.NA] * len(result), dtype='boolean')
    correct[paired] = true_labels[paired] == pred_labels[paired]
    result['Class Correct'] = correct
    result['Classification Rule'] = RULE_VERSION
    return result


def export_classification(frame, output_dir, dataset=None, *, default_catalyst=None,
                          group_columns=(), stem='classification', role='Regression predictions',
                          plots=True, dpi=600):


    from .regression_plotting import (interval_metrics, dump_json, prediction_report,
                                      save_figure, STYLE, plt)
    result = attach_classification(frame, dataset, default_catalyst)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    group_columns = list(dict.fromkeys(['Classification Dataset', *group_columns]))
    group_columns = [c for c in group_columns if c in result]
    rows, reports = [], []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, part in result.groupby(grouper, sort=False, dropna=False):
        key = (key,) if len(group_columns) == 1 else key
        identity = dict(zip(group_columns, key))
        metal = identity['Classification Dataset']
        cuts = BOUNDARIES[metal]
        pred_col = _column(part, (PREDICTED_G, 'predicted_delta_g_kcal_mol'), True)
        true_col = _column(part, (TRUE_G, 'true_delta_g_kcal_mol', 'G (kcal/mol)'))
        valid = part['Predicted Class'].notna()
        paired = valid & part['True Class'].notna()
        metrics = (interval_metrics(part.loc[paired, true_col].to_numpy(float),
                                    part.loc[paired, pred_col].to_numpy(float), cuts)
                   if paired.any() else None)
        coverage = {'total_rows': len(part), 'valid_predictions': int(valid.sum()),
                    'scored_rows': int(paired.sum()), 'failed_predictions': int((~valid).sum()),
                    'unscored_predictions': int((valid & ~paired).sum())}
        rows.append({**identity, **coverage, 'lower_boundary': cuts[0], 'upper_boundary': cuts[1],
                     **({k: metrics[k] for k in ('correct', 'accuracy', 'balanced_accuracy', 'macro_f1')}
                        if metrics else {'correct': None, 'accuracy': None, 'balanced_accuracy': None, 'macro_f1': None})})
        reports.append({'identity': identity, 'coverage': coverage, 'metrics': metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / f'{stem}_metrics.csv', index=False, encoding='utf-8-sig')
    dump_json(output / f'{stem}_metrics.json', {'rule': RULE_VERSION, 'boundaries': BOUNDARIES,
              'role': role, 'equality': 'higher class', 'groups': reports})
    if plots and len(rows) == 1:
        plotting = result.copy()
        plotting[PREDICTED_G] = plotting[_column(result, (PREDICTED_G, 'predicted_delta_g_kcal_mol'), True)]
        true_col = _column(result, (TRUE_G, 'true_delta_g_kcal_mol', 'G (kcal/mol)'))
        if true_col:
            plotting[TRUE_G] = plotting[true_col]
        prediction_report(plotting, output / 'figures', boundaries=BOUNDARIES[rows[0]['Classification Dataset']],
                          title=role, role_note='Fixed G intervals applied after regression; no classifier training.',
                          stem=stem, dpi=dpi)
    elif plots and len(rows) > 1:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(11, max(4, len(rows) * .30 + 1.5)))
            scored = summary[summary.accuracy.notna()].sort_values('accuracy')
            labels = [' | '.join(str(r[c]) for c in group_columns) for _, r in scored.iterrows()]
            bars = ax.barh(labels, scored.accuracy, color='#267A9E')
            for bar, value in zip(bars, scored.accuracy):
                ax.text(value + .01, bar.get_y() + bar.get_height()/2, f'{value:.2%}', va='center', fontsize=9)
            ax.set(xlim=(0, 1.1), xlabel='Three-class accuracy', title=role)
            fig.text(.02, .01, 'Fixed G boundaries (kcal/mol): Ir 23 / 25; Co 24 / 26. Equality enters the higher class.', fontsize=9)
            fig.tight_layout(rect=(0, .03, 1, 1))
            save_figure(fig, output / 'figures', stem, dpi)
    return result
