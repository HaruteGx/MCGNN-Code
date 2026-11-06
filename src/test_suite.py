import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/Co-transfer"))
sys.path.insert(0, str(ROOT / "src/Co-transfer/Cotrans"))

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from src.regression_classification import attach_classification, export_classification


def test_mixed_metals_threshold_equality_and_failed_prediction():
    frame = pd.DataFrame({'Cat': ['Ir', 'Ir', 'Ir', 'Co', 'Co', 'Co'],
                          'Predicted G (kcal/mol)': [22.99, 23, 25, 23.99, 24, 26],
                          'G (kcal/mol)': [23, 25, 25, 24, 26, 26],
                          'status': ['OK'] * 5 + ['SKIP']})
    result = attach_classification(frame)
    assert result['Predicted Class'].iloc[:5].tolist() == [0, 1, 2, 0, 1]
    assert pd.isna(result['Predicted Class'].iloc[5])
    assert pd.isna(result['Class Correct'].iloc[5])
    assert result['True Class'].tolist() == [1, 2, 2, 1, 2, 2]
    pd.testing.assert_series_equal(result['Predicted G (kcal/mol)'], frame['Predicted G (kcal/mol)'])
    assert 'Predicted Class' not in frame


def test_missing_or_unknown_metal_requires_explicit_selection():
    frame = pd.DataFrame({'Cat': ['Pt'], 'Predicted G (kcal/mol)': [25.]})
    with pytest.raises(ValueError, match='Ir/Co'):
        attach_classification(frame)
    assert attach_classification(frame, 'Co')['Predicted Class'].tolist() == [1]


def test_no_experimental_value_does_not_create_accuracy(tmp_path):
    frame = pd.DataFrame({'Predicted G (kcal/mol)': [24., np.nan, 26.]})
    result = export_classification(frame, tmp_path, 'Co', plots=True, dpi=72)
    assert result['True Class'].isna().all()
    report = json.loads((tmp_path/'classification_metrics.json').read_text())['groups'][0]
    assert report['metrics'] is None
    assert report['coverage'] == {'total_rows': 3, 'valid_predictions': 2, 'scored_rows': 0,
                                  'failed_predictions': 1, 'unscored_predictions': 2}
    assert (tmp_path/'figures/classification.png').is_file()


def test_direct_g_baselines_use_same_rule_and_report_partial_coverage(tmp_path):
    frame = pd.DataFrame({'dataset': ['Co'] * 4, 'model': ['RF'] * 4,
        'true_delta_g_kcal_mol': [23., 24., 26., 27.],
        'predicted_delta_g_kcal_mol': [24., 24., 26., np.nan]})
    result = export_classification(frame, tmp_path, group_columns=('model',), plots=False)
    report = json.loads((tmp_path/'classification_metrics.json').read_text())['groups'][0]
    assert report['metrics']['confusion'] == [[0, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert report['metrics']['accuracy'] == pytest.approx(2/3)
    assert report['coverage']['failed_predictions'] == 1
    assert result['Predicted Class'].iloc[:3].tolist() == [1, 1, 2]


def test_saved_summary_rejects_failed_prediction_even_when_g_is_finite(tmp_path, monkeypatch):
    import importlib.util

    entry = Path(__file__).resolve().parents[1] / 'scripts/summarize_regression_classification.py'
    spec = importlib.util.spec_from_file_location('summary_entry_for_test', entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'predictions.csv'
    pd.DataFrame({'Name': ['sample'], 'G (kcal/mol)': [24.],
                  'Predicted G (kcal/mol)': [24.], 'status': ['SKIP'],
                  'reason': ['invalid input']}).to_csv(path, index=False)
    monkeypatch.setattr(module, 'sources', lambda root: [(path, 'Co', 'Regress', 'grouped_oof')])
    with pytest.raises(ValueError, match='incomplete or failed predictions'):
        module.run(root=tmp_path, plots=False)
    assert not (tmp_path / 'outputs/regression_classification/summary_metrics.csv').exists()


def test_saved_summary_combines_legacy_and_explicit_success_status(tmp_path, monkeypatch):
    import importlib.util

    entry = Path(__file__).resolve().parents[1] / 'scripts/summarize_regression_classification.py'
    spec = importlib.util.spec_from_file_location('summary_mixed_status_for_test', entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ir_path, co_path = tmp_path / 'ir.csv', tmp_path / 'co.csv'
    pd.DataFrame({'Name': ['ir'], 'G (kcal/mol)': [24.],
                  'Predicted G (kcal/mol)': [24.]}).to_csv(ir_path, index=False)
    pd.DataFrame({'Name': ['co'], 'G (kcal/mol)': [26.],
                  'Predicted G (kcal/mol)': [26.], 'status': ['OK'],
                  'Fold': [1]}).to_csv(co_path, index=False)
    monkeypatch.setattr(module, 'sources', lambda root: [
        (ir_path, 'Ir', 'Regress', 'existing_test'),
        (co_path, 'Co', 'Regress transfer', 'grouped_oof'),
    ])
    module.run(root=tmp_path, plots=False)
    result = pd.read_csv(tmp_path / 'outputs/regression_classification/all_predictions.csv')
    assert result['Predicted Class'].tolist() == [1, 2]
    assert result['status'].eq('OK').all()

if os.environ.get("MCGNN_TEST_MODE") != "lightweight":
    from pathlib import Path
    import sys

    import numpy as np
    import pytest
    import torch

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT))

    from analysis.atom_bond_importance.attribution import (
        integrated_gradients, substrate_mapping, reference_features, explain,
    )
    from src.graph_builders import create_reactant_graph_with_metal
    from src.mcencoding import encode_catalyst_feature
    from src.models import ReactionEnergyPredictionModel
    from src.features import REACTION_SITE_NODE_FEATURE_INDEX, NODE_FEATURE_NAMES


    def test_ig_analytic_interaction_and_signed_completeness():
        x = (torch.tensor([2., 3.]), torch.tensor([4.]))
        zero = tuple(torch.zeros_like(v) for v in x)
        def score(a, b):
            return 2*a[0]-3*a[1]+a[0]*b[0]+7
        attribution, audit = integrated_gradients(score, x, zero, 4, 8, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(attribution[0], [8., -9.], atol=1e-5)
        np.testing.assert_allclose(attribution[1], [4.], atol=1e-5)
        assert audit["converged"] and abs(audit["attribution_sum"]-3) < 1e-5


    def test_constant_unused_input_has_zero_attribution():
        x = (torch.tensor([2.]), torch.tensor([5.]))
        attribution, audit = integrated_gradients(lambda a, b: a.sum()*0+8, x,
                                                  tuple(torch.zeros_like(v) for v in x), 4, 8)
        assert audit["converged"]
        assert all(torch.count_nonzero(a) == 0 for a in attribution)


    def make_graph():
        graph = create_reactant_graph_with_metal("CCO", "B", "N", encode_catalyst_feature("Ir"),
                                                 [1], "CCO.B.N", torch.device("cpu"))
        graph.ndata["global"] = torch.ones(graph.num_nodes(), 4)
        return graph


    def test_mapping_baseline_keeps_roles_non_substrate_and_virtual_edges():
        graph = make_graph()
        mol, pairs = substrate_mapping(graph, "CCO")
        original, base = reference_features(graph, mol, pairs)
        assert len(pairs) == 2
        torch.testing.assert_close(base[0][3:], original[0][3:])
        torch.testing.assert_close(base[0][:, REACTION_SITE_NODE_FEATURE_INDEX], original[0][:, REACTION_SITE_NODE_FEATURE_INDEX])
        for col, name in enumerate(NODE_FEATURE_NAMES):
            if name.endswith("_available"):
                torch.testing.assert_close(base[0][:, col], original[0][:, col])
        chemical = {eid for pair in pairs for eid in pair}
        rest = [eid for eid in range(graph.num_edges()) if eid not in chemical]
        torch.testing.assert_close(base[1][rest], original[1][rest])
        for pair in pairs:
            torch.testing.assert_close(base[1][pair[0]], base[1][pair[1]])
        with pytest.raises(AssertionError):
            substrate_mapping(graph, "CCN")


    def test_real_gnn_directions_sum_and_graph_model_stay_unchanged():
        torch.manual_seed(16)
        torch.set_num_threads(1)
        model = ReactionEnergyPredictionModel(4, environment_global_feature_dim=4,
                                             reactant_hidden_feats=8, reactant_out_feats=8,
                                             product_hidden_feats=8, product_out_feats=8,
                                             fc_hidden_feats=8, catalyst_context_dim=4).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        graph = make_graph()
        nodes, edges = graph.ndata["node"].clone(), graph.edata["edge"].clone()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        sample = ("test", graph.clone(), graph, graph.clone(), 300., 24.)
        mol, pairs, atoms, bonds, node, edge, audit = explain(model, sample, "CCO", 16, 256)
        assert audit["converged"]
        np.testing.assert_allclose(bonds, [edge[pair].sum() for pair in pairs])
        np.testing.assert_allclose(atoms.sum()+bonds.sum(), audit["attribution_sum"], atol=2e-5)
        torch.testing.assert_close(nodes, graph.ndata["node"])
        torch.testing.assert_close(edges, graph.edata["edge"])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(weights[key], value)

    from pathlib import Path
    import sys

    import numpy as np
    import pytest
    import torch

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src/Co-transfer/Cotrans"))
    from joint_objective import JointClassificationRegressionLoss, classification_weight_at_epoch, selection_key


    def test_zero_classification_weight_is_exact_normalized_mse():
        pred, true = torch.tensor([22., 25., 28.]), torch.tensor([23., 24., 27.])
        loss = JointClassificationRegressionLoss(classification_weight=0, regression_scale=2)
        torch.testing.assert_close(loss(pred, true), ((pred - true) / 2).square().mean())


    def test_joint_penalties_and_gradient_match_sum():
        pred = torch.tensor([22., 24., 26., 28.], dtype=torch.float64, requires_grad=True)
        true = torch.tensor([23., 24., 26., 27.], dtype=torch.float64)
        loss = JointClassificationRegressionLoss(classification_weight=.5).double()
        cls, reg = loss.per_sample(pred, true)
        value = loss(pred, true)
        torch.testing.assert_close(value, reg.mean() + .5 * cls.mean())
        value.backward()
        z = (pred.detach()[:, None] - torch.tensor([24., 26.])) / .5
        labels = (true[:, None] >= torch.tensor([24., 26.])).double()
        expected = (2 * (pred.detach() - true) + .5 * (z.sigmoid() - labels).mean(dim=1) / .5) / len(pred)
        torch.testing.assert_close(pred.grad, expected)


    def test_regression_anchor_penalizes_drift_even_with_correct_classes():
        criterion = JointClassificationRegressionLoss()
        true = torch.tensor([23., 27.])
        close = criterion(torch.tensor([22.9, 27.1]), true)
        far = criterion(torch.tensor([10., 40.]), true)
        assert far > close


    def test_joint_selection_rejects_accuracy_gain_with_worse_combined_loss():
        balanced = {"accuracy": .76, "macro_f1": .75, "total_loss": 1.0}
        drifted = {"accuracy": .88, "macro_f1": .87, "total_loss": 2.0}
        assert selection_key(balanced) > selection_key(drifted)


    def test_warmup_is_bounded_and_can_continue_across_stages():
        assert classification_weight_at_epoch(.5, 1, 10) == .05
        assert classification_weight_at_epoch(.5, 10, 10) == .5
        assert classification_weight_at_epoch(.5, 100, 10) == .5
        assert classification_weight_at_epoch(.5, 1, 0) == .5


    @pytest.mark.parametrize("kwargs", [{"classification_weight": -1}, {"regression_scale": 0}, {"temperature": float('nan')}])
    def test_invalid_joint_loss_settings_fail(kwargs):
        with pytest.raises(ValueError):
            JointClassificationRegressionLoss(**kwargs)


    def test_training_logs_components_and_uses_fixed_validation_weight(monkeypatch):
        import joint_training as training
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.offset = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, environment, reactant, product, temperature):
                return environment + self.offset
        def collate(samples, device):
            names, x, y = zip(*samples)
            return names, torch.tensor(x), None, None, None, torch.tensor(y)
        monkeypatch.setattr(training, "collate_regression", collate)
        data = [("a", 22., 23.), ("b", 24., 25.), ("c", 27., 27.)]
        model = Tiny()
        result = training.train_joint(model, data, data, torch.device('cpu'), epochs=2, learning_rate=.01,
                                       batch_size=3, warmup_epochs=10, classification_weight=.5)
        for row in result['history']:
            assert row['train_loss'] == pytest.approx(row['train_regression_loss'] + row['classification_weight'] * row['train_classification_loss'])
            assert row['validation_total_loss'] == pytest.approx(row['validation_regression_loss'] + .5 * row['validation_classification_loss'])
        best = max(result['history'], key=lambda row: (-row['validation_total_loss'], row['validation_accuracy'], row['validation_macro_f1']))
        assert result['best_epoch'] == best['epoch']
        restored = training.score_predictions([23., 25., 27.], (torch.tensor([22., 24., 27.]) + model.offset).detach().numpy(), JointClassificationRegressionLoss())
        assert restored['total_loss'] == pytest.approx(result['best_metrics']['total_loss'], abs=1e-6)

    import pytest
    import torch
    import torch.nn as nn

    from src.models import (
        ReactionEnergyPredictionModel,
        regression_model_config_from_checkpoint,
    )


    class ConstantOutput(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, *args, **kwargs):
            return self.value


    class ConstantReactant(nn.Module):
        def forward(self, graph, return_node_states=False):
            pooled = torch.ones((1, 3))
            nodes = torch.ones((2, 3))
            return (pooled, nodes) if return_node_states else pooled


    def model_with_constant_branches(ablate_branches=()):
        model = ReactionEnergyPredictionModel(
            global_feature_dim=4,
            environment_global_feature_dim=4,
            reactant_hidden_feats=4,
            reactant_out_feats=3,
            product_hidden_feats=4,
            product_out_feats=2,
            fc_hidden_feats=5,
            catalyst_context_dim=2,
            ablate_branches=ablate_branches,
        )
        model.environment_gnn = ConstantOutput(torch.ones((1, 128)))
        model.reactant_gnn = ConstantReactant()
        model.product_gnn = ConstantOutput(torch.ones((1, 2)))
        model.catalyst_context = ConstantOutput(torch.ones((1, 2)))
        model.fc1 = nn.Identity()
        return model


    @pytest.mark.parametrize(
        ("branches", "expected_zero_slices"),
        [
            (("environment",), (slice(0, 128),)),
            (("reactant",), (slice(128, 131), slice(133, 135))),
            (("product",), (slice(131, 133),)),
            (("environment", "reactant"), (slice(0, 131), slice(133, 135))),
            (("environment", "product"), (slice(0, 128), slice(131, 133))),
            (("reactant", "product"), (slice(128, 135),)),
        ],
    )
    def test_graph_ablation_zeros_only_its_fusion_route(branches, expected_zero_slices):
        model = model_with_constant_branches(branches)
        encoded = model.encode_reaction(None, None, None)

        assert encoded.shape == (1, 135)
        expected = torch.ones((1, 135))
        for section in expected_zero_slices:
            expected[:, section] = 0
        assert torch.equal(encoded, expected)


    def test_default_model_preserves_every_graph_route():
        model = model_with_constant_branches()
        encoded = model.encode_reaction(None, None, None)

        assert encoded.shape == (1, 135)
        assert torch.all(encoded == 1)


    def test_unknown_graph_ablation_is_rejected():
        with pytest.raises(ValueError, match="Unknown graph branches"):
            model_with_constant_branches(["solvent"])


    @pytest.mark.parametrize("branches", [
        ("reactant",),
        ("environment", "reactant"),
        ("environment", "product"),
        ("reactant", "product"),
    ])
    def test_checkpoint_metadata_restores_ablation_configuration(branches):
        model = ReactionEnergyPredictionModel(
            global_feature_dim=4,
            environment_global_feature_dim=4,
            reactant_hidden_feats=4,
            reactant_out_feats=3,
            product_hidden_feats=4,
            product_out_feats=2,
            fc_hidden_feats=5,
            catalyst_context_dim=2,
            ablate_branches=branches,
        )
        config = regression_model_config_from_checkpoint(
            model.state_dict(), {"model_config": {"ablate_branches": list(branches)}}
        )

        restored = ReactionEnergyPredictionModel(**config)
        assert restored.ablate_branches == frozenset(branches)

    import importlib.util
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pytest
    import torch
    from rdkit import Chem

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.data_splits import (
        audit_disjoint_inputs, checkpoint_split_config, derive_ir_test_set,
        load_frozen_ir_split, sha256_file, derive_random_ir_split, ensure_random_ir_split,
    )
    from src.ir_protocol import ir_run_paths


    @pytest.fixture(scope="module")
    def tables():
        paths = ir_run_paths(ROOT)
        return (pd.read_csv(ROOT / "data/Database Ir.csv"),
                pd.read_csv(paths["development"]), pd.read_csv(paths["test"]))


    def test_current_split_has_exactly_251_preserved_records(tables):
        full, development, saved = tables
        expected = full[~full.Name.isin(development.Name)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, expected)
        assert len(saved) == 251 and len(development) == 2251
        assert list(saved.columns) == list(full.columns)
        assert set(saved.Name) | set(development.Name) == set(full.Name)


    def test_target_difference_does_not_move_a_training_record_to_test(tables):
        full, development, saved = tables
        reduced_train = development.iloc[:2].copy()
        reduced_full = pd.concat([reduced_train, saved.iloc[:1]], ignore_index=True)
        reduced_full.loc[0, "G (kcal/mol)"] += 9.0
        result, report = derive_ir_test_set(reduced_full, reduced_train)
        assert result.Name.tolist() == saved.Name.iloc[:1].tolist()
        assert len(report["shared_value_differences"]) == 1


    def test_duplicate_names_or_non_subset_are_rejected(tables):
        full, development, _ = tables
        with pytest.raises(ValueError, match="unique"):
            derive_ir_test_set(pd.concat([full, full.iloc[:1]]), development)
        invalid = development.iloc[:1].copy()
        invalid.loc[invalid.index[0], "Name"] = "not-in-source"
        with pytest.raises(ValueError, match="missing"):
            derive_ir_test_set(full, invalid)


    def test_different_name_and_equivalent_smiles_do_not_bypass_overlap_guard(tables):
        _, development, _ = tables
        duplicate = development.iloc[:1].copy()
        duplicate.loc[duplicate.index[0], "Name"] = "new-name-same-input"
        duplicate.loc[duplicate.index[0], "L-type"] = Chem.MolToSmiles(
            Chem.MolFromSmiles(duplicate.iloc[0]["L-type"]), canonical=True)
        with pytest.raises(ValueError, match="canonical model-input overlap"):
            audit_disjoint_inputs(development, duplicate)


    def test_frozen_split_detects_changes_to_test_labels(tmp_path, tables):
        paths = ir_run_paths(ROOT)
        development_path = paths["development"]
        manifest_path = paths["manifest"]
        development, test, manifest = load_frozen_ir_split(
            development_path, paths["test"], manifest_path, full_path=paths["full"])
        assert checkpoint_split_config(manifest)["test_used_for_model_selection"] is False
        changed = test.copy()
        changed.loc[0, "G (kcal/mol)"] += 1
        changed_path = tmp_path / "changed.csv"
        changed.to_csv(changed_path, index=False)
        with pytest.raises(ValueError, match="test data changed"):
            load_frozen_ir_split(development_path, changed_path, manifest_path, full_path=paths["full"])


    def test_five_fold_entrypoint_evaluates_test_only_after_selection(tmp_path, monkeypatch, tables):


        torch.set_num_threads(2)
        full = tables[0].iloc[:30].copy()
        full_path = tmp_path / "Database Ir.csv"
        full.to_csv(full_path, index=False)
        dev_path, test_path = tmp_path / "development.csv", tmp_path / "test.csv"
        manifest_path = tmp_path / "split.json"
        development, test, manifest = ensure_random_ir_split(
            full_path, dev_path, test_path, manifest_path, seed=10, test_fraction=0.10)
        spec = importlib.util.spec_from_file_location("ir_training_test_entry", ROOT / "scripts/train_full_model.py")
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        monkeypatch.setattr(training, "DATA_PATH", str(full_path))
        monkeypatch.setattr(training, "DEVELOPMENT_DATA_PATH", str(dev_path))
        monkeypatch.setattr(training, "TEST_DATA_PATH", str(test_path))
        monkeypatch.setattr(training, "SPLIT_MANIFEST_PATH", str(manifest_path))
        monkeypatch.setattr(training, "OUTPUT_DIR", str(tmp_path / "run"))
        monkeypatch.setattr(training, "STABLE_MODEL_PATH", str(tmp_path / "model.pth"))
        monkeypatch.setattr(training, "get_device", lambda **kwargs: torch.device("cpu"))
        monkeypatch.setattr(training, "FINGERPRINT_NBITS", 32)
        monkeypatch.setattr(training, "INCLUDE_3D_DESCRIPTORS", False)
        monkeypatch.setattr(training, "NUM_EPOCHS", 1)
        monkeypatch.setattr(training, "NUM_STRATIFICATION_BINS", 2)
        monkeypatch.setattr(training, "MODEL_CONFIG", dict(
            dropout=0.0, reactant_hidden_feats=16, reactant_out_feats=12,
            product_hidden_feats=16, product_out_feats=12, fc_hidden_feats=16,
            catalyst_context_dim=8))
        monkeypatch.setattr(training, "plot_training_history", lambda *a, **k: None)
        monkeypatch.setattr(training, "save_regression_scatter", lambda *a, **k: None)
        holdout_module = sys.modules["src.holdout_eval"]
        monkeypatch.setattr(holdout_module, "save_regression_scatter", lambda *a, **k: None)
        train_eval = sys.modules["src.train_eval"]
        monkeypatch.setattr(train_eval, "save_regression_scatter", lambda *a, **k: None)
        events = []
        test_ids = set(test.Name)
        actual_fit, actual_test = training.train_regressor, training.evaluate_frozen_holdout

        def fit(**kwargs):
            names = {row[0] for row in kwargs["train_data"]+kwargs["validation_data"]}
            assert names == set(development.Name) and not names & test_ids
            events.append("fit")
            return actual_fit(**kwargs)

        def validate(model, split, device, save_path, title, batch_size, target_column):
            return train_eval.evaluate_and_save(model, split, device, save_path, title,
                                                batch_size, target_column)

        def evaluate(checkpoint, frame, split, device, output, batch_size):
            assert events == ["fit"]*5
            assert frame.Name.tolist() == test.Name.tolist()
            before = sha256_file(checkpoint)
            report, results = actual_test(checkpoint, frame, split, device, output, batch_size)
            assert sha256_file(checkpoint) == before
            assert report["holdout_split"]["test_used_for_model_selection"] is False
            events.append("test")
            return report, results

        monkeypatch.setattr(training, "train_regressor", fit)
        monkeypatch.setattr(training, "evaluate_and_save", validate)
        monkeypatch.setattr(training, "evaluate_frozen_holdout", evaluate)
        training.main()
        assert events == ["fit"]*5+["test"]
        report = json.loads((tmp_path / "run/test/test_metrics.json").read_text(encoding="utf-8"))
        assert report["test_rows"] == 3
        with pd.ExcelFile(tmp_path / "run/five_fold_cross_validation_summary.xlsx") as workbook:
            assert "Independent Test Metrics" in workbook.sheet_names
            assert "Independent Test Predictions" in workbook.sheet_names

        from src.completed_run_audit import verify_completed_class_run
        selected = tmp_path / "run/best_ir_xtb_metal_regression_checkpoint.pth"
        with pytest.raises(ValueError, match="cannot be reused"):
            verify_completed_class_run(tmp_path / "run", selected, dev_path, manifest)
        model, saved_metadata, config = holdout_module.load_regression_checkpoint(selected, torch.device("cpu"))
        saved_metadata.pop("holdout_split")
        monkeypatch.setattr(holdout_module, "load_regression_checkpoint", lambda *a: (model, saved_metadata, config))
        with pytest.raises(ValueError, match="not trained under this frozen"):
            actual_test(selected, test, manifest, torch.device("cpu"), tmp_path / "legacy_test",
                        completed_run_dir=tmp_path / "run", development_path=dev_path)
        saved_metadata["holdout_split"] = checkpoint_split_config(manifest)
        saved_metadata["holdout_split"]["test_split_seed"] = 11
        with pytest.raises(ValueError, match="not trained under this frozen"):
            actual_test(selected, test, manifest, torch.device("cpu"), tmp_path / "wrong_seed")


    def test_holdout_evaluation_rejects_legacy_protocol(monkeypatch, tables, tmp_path):
        from src import holdout_eval
        _, _, test = tables
        manifest = json.loads(ir_run_paths(ROOT)["manifest"].read_text(encoding="utf-8"))
        monkeypatch.setattr(holdout_eval, "load_regression_checkpoint", lambda *a: (None, {}, {}))
        with pytest.raises(ValueError, match="not trained under this frozen"):
            holdout_eval.evaluate_frozen_holdout("old.pth", test, manifest, torch.device("cpu"), tmp_path)


    def test_random_90_10_matches_seed_and_preserves_full_source(tables):
        full = tables[0]
        paths = ir_run_paths(ROOT)
        development, test, manifest = load_frozen_ir_split(
            paths["development"], paths["test"], paths["manifest"], full_path=paths["full"])
        permutation = np.random.RandomState(10).permutation(len(full))
        expected_test_indices = np.sort(permutation[:251])
        pd.testing.assert_frame_equal(test, full.iloc[expected_test_indices].reset_index(drop=True))
        assert len(test) == 251 and len(development) == 2251
        assert not set(development.Name) & set(test.Name)
        assert set(development.Name) | set(test.Name) == set(full.Name)
        assert manifest["test_stratified"] is False
        assert manifest["audit"]["overlapping_model_input_test_rows"] == 12
        combined = pd.concat([development, test])
        assert combined.set_index("Name").loc["LRM106", "G (kcal/mol)"] == 29.7437


    def test_random_split_is_reproducible_and_independent_of_labels_and_global_seed(tables):
        full = tables[0].iloc[:40].copy()
        np.random.seed(999)
        first = derive_random_ir_split(full, seed=10)[1].Name.tolist()
        np.random.seed(63)
        full["G (kcal/mol)"] = np.arange(len(full))*100
        second = derive_random_ir_split(full, seed=10)[1].Name.tolist()
        different = derive_random_ir_split(full, seed=11)[1].Name.tolist()
        assert first == second and first != different
        assert ir_run_paths(ROOT, seed=10)["output"] != ir_run_paths(ROOT, seed=11)["output"]


    def test_random_protocol_does_not_silently_group_or_drop_similar_records(tables):
        full = tables[0].iloc[:10].copy()


        for column in full.columns:
            if column != "Name":
                full[column] = full.iloc[0][column]
        development, test, manifest = derive_random_ir_split(full, seed=10)
        assert len(development) == 9 and len(test) == 1
        assert manifest["audit"]["canonical_model_input_overlap"] == 1


    def test_random_split_reuse_and_source_or_seed_drift_are_guarded(tmp_path, tables):
        full_path = tmp_path / "full.csv"
        tables[0].iloc[:30].to_csv(full_path, index=False)
        dev, test, manifest = tmp_path / "dev.csv", tmp_path / "test.csv", tmp_path / "split.json"
        first = ensure_random_ir_split(full_path, dev, test, manifest, seed=10)
        hashes = tuple(sha256_file(path) for path in (dev, test, manifest))
        second = ensure_random_ir_split(full_path, dev, test, manifest, seed=10)
        assert hashes == tuple(sha256_file(path) for path in (dev, test, manifest))
        assert first[2]["test_names"] == second[2]["test_names"]
        with pytest.raises(ValueError, match="different protocol"):
            ensure_random_ir_split(full_path, dev, test, manifest, seed=11)
        changed = pd.read_csv(full_path)
        changed.loc[0, "G (kcal/mol)"] += 1
        changed.to_csv(full_path, index=False)
        with pytest.raises(ValueError, match="Full Ir source"):
            ensure_random_ir_split(full_path, dev, test, manifest, seed=10)

    from pathlib import Path
    import importlib.util
    import json
    import os
    import sys

    import pandas as pd
    import pytest

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'src/Co-transfer'))


    def load_entry(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


    def test_frozen_predictions_load_from_another_working_directory(tmp_path, monkeypatch):
        from src.ir_protocol import load_ir_test_predictions, ir_run_paths
        test = pd.read_csv(ir_run_paths(ROOT)['test'])
        monkeypatch.chdir(tmp_path)
        result = load_ir_test_predictions(ROOT, test)
        assert len(result) == 251
        assert result.index.tolist() == test.Name.astype(str).tolist()


    def test_reference_rejects_changed_test_labels():
        from src.ir_protocol import load_ir_test_predictions, ir_run_paths
        test = pd.read_csv(ir_run_paths(ROOT)['test'])
        test.loc[0, 'G (kcal/mol)'] += 1
        with pytest.raises(AssertionError):
            load_ir_test_predictions(ROOT, test)


    def test_split_preparation_does_not_require_removed_data_manifest(tmp_path, monkeypatch):
        module = load_entry('public_prepare_split', 'scripts/prepare_ir_test_set.py')
        monkeypatch.setattr(module, 'PROJECT_DIR', tmp_path)
        monkeypatch.setattr(module, 'ensure_random_ir_split', lambda *a, **k: (
            pd.DataFrame({'Name': ['dev']}), pd.DataFrame({'Name': ['test']}),
            {'full_rows': 2, 'audit': {'overlapping_model_input_test_rows': 0}}))
        module.main([])
        assert not (tmp_path / 'data/MANIFEST.csv').exists()


    def test_ml_results_use_actual_paths():
        for name in ('ir_ml_metrics.csv', 'ir_ml_predictions.csv'):
            assert (ROOT / 'outputs/baselines/ML' / name).is_file()


    def test_ablation_runner_and_summary_use_same_flat_directory(tmp_path):
        from baselines.ablation.run_graph_ablations import commands_for
        from baselines.ablation.summarize_graph_ablations import expected_files, DEFAULT_REFERENCE
        argv = commands_for('environment', tmp_path, skip_co=True, ir_epochs=1,
                            stage1_epochs=1, stage2_epochs=1, cpu=True)[0][1]
        assert Path(argv[argv.index('--output-dir')+1]) == tmp_path / 'environment'
        assert expected_files(tmp_path, 'environment', include_co=False)[0][2] == (
            tmp_path / 'environment/five_fold_oof_predictions.csv')
        assert DEFAULT_REFERENCE.is_file()


    def test_atomic_json_write_uses_requested_destination(tmp_path):
        from co_transfer_holdout.protocol import dump
        destination = tmp_path / 'nested/metadata/new.json'
        dump(destination, {'value': 7})
        assert json.loads(destination.read_text()) == {'value': 7}
        assert not destination.with_suffix('.json.tmp').exists()


    def test_permutation_output_uses_current_directory():
        module = load_entry('public_importance_args', 'analysis/feature_importance/run_ir_importance.py')
        assert module.parse_args(['--smoke']).output_dir.parent == (
            ROOT / 'outputs/interpretability/feature_importance')


    def test_relocated_ir_ablation_tables_include_full_model(tmp_path):
        import shutil
        from baselines.ablation.summarize_graph_ablations import expected_files, summarize
        source = ROOT / 'outputs/interpretability/graph_ablations'
        for branch in ('environment', 'reactant', 'product'):
            for (_, _, old), (_, _, new) in zip(
                    expected_files(source, branch, include_co=False),
                    expected_files(tmp_path, branch, include_co=False)):
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2((old), new)
        summary, _, missing = summarize(tmp_path, include_co=False)
        assert not missing
        assert len(summary) == 8
        assert set(summary.ablation) == {'full_model', 'environment', 'reactant', 'product'}

    import copy
    import json
    import shutil
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pytest
    import torch
    from rdkit import Chem, RDLogger

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from src.dataset import build_dataset, filter_valid_indices
    from src.descriptors import global_descriptor_config, environment_global_feature_dimension
    from src.features import NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
    from src.models import ReactionEnergyPredictionModel, build_regression_model_from_checkpoint
    from src.prediction import predict_dataframe, load_regression_checkpoint
    from src.train_eval import collate_regression, train_regressor
    from src.xtb_features import XTBFeatureStore, xtb_feature_config, get_xtb_store

    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.warning')


    @pytest.fixture(scope="module")
    def store(tmp_path_factory):
        directory = tmp_path_factory.mktemp("xtb_source")
        source = ROOT / "data" / "xtb_ir_features"
        manifest = pd.read_csv(source / "xtb_unique_molecule_features.csv")
        selected = manifest[manifest.job_id.isin(["39ba84e13e6e6f35", "d166861a900c92f8", "0573eb1197f452b7"])]
        selected.to_csv(directory / "xtb_unique_molecule_features.csv", index=False)
        for row in selected.itertuples():
            dest = directory / "calculations" / row.role / row.job_id
            dest.mkdir(parents=True)
            for name in ("result.json", "xtbout.json", "xtbopt.xyz", "wbo", "sp.out", "vibspectrum"):
                shutil.copyfile(source / "calculations" / row.role / row.job_id / name, dest / name)
        return XTBFeatureStore(directory, directory / "cache")


    def example_smiles():
        return "O=C(N(C(OC(C)(C)C)=O)C(OC(C)(C)C)=O)C1=CC=CC=C1"


    def test_co_xtb_store_is_separate_and_parseable(tmp_path):
        root = ROOT / "data" / "xtb_Co_features"
        manifest = pd.read_csv(root / "xtb_unique_molecule_features.csv")
        assert set(manifest["role"]) == {"sub", "prod", "lig", "boron"}
        assert manifest["status"].eq("ok").all()
        store = XTBFeatureStore(root, tmp_path / "co_cache")
        for role in ("sub", "prod", "lig", "boron"):
            row = manifest.loc[manifest["role"].eq(role)].iloc[0]
            features = store.get(row["smiles"], role)
            assert features.available, (role, features.reason)
            assert np.isfinite(features.atom).all()
            assert np.isfinite(features.global_vector).all()
        report = store.report()
        assert Path(report["root"]) == root.resolve()
        assert report["manifest_sha256"] != XTBFeatureStore().manifest_sha256


    def test_raw_frontier_dipole_charge_and_wbo(store):
        result = store.get(example_smiles(), "sub")
        assert result.available, result.reason
        assert result.global_vector[0]*10 == pytest.approx(-11.17471542, abs=2e-6)
        assert result.global_vector[1]*10 == pytest.approx(-7.56152350, abs=2e-6)
        assert result.global_vector[2]*10 == pytest.approx(5.60433572, abs=2e-6)
        assert result.atom[20, 0] == pytest.approx(-0.02052307, abs=1e-7)
        edge = result.edge_vector(0, 1)
        assert edge[0] == pytest.approx(1.80334572, abs=2e-6)
        assert edge[3:5] == [1.0, 1.0]
        assert result.edge_vector(1, 0) == edge
        assert result.atom[20, 7] == 1
        assert 0.8 < result.atom[20, 5] < 1.3


    def test_equivalent_smiles_maps_atom_properties(store):
        mol = Chem.MolFromSmiles(example_smiles())
        query = Chem.MolToSmiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False)
        mapped = store.get(query, "sub")
        query_mol = Chem.MolFromSmiles(query)
        mapping = mol.GetSubstructMatch(query_mol, useChirality=True)
        original = store.get(example_smiles(), "sub")
        assert mapped.available
        assert query != example_smiles()
        np.testing.assert_allclose(mapped.atom, original.atom[list(mapping)])
        for bond in query_mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            np.testing.assert_allclose(mapped.edge_vector(a, b), original.edge_vector(mapping[a], mapping[b]))


    def test_thermally_occupied_lumo_uses_xtb_frontier_labels(store):
        row = next(value[1] for value in store.jobs.values() if value[1]["job_id"] == "0573eb1197f452b7")
        result = store.get(row["smiles"], "prod")
        assert result.available, result.reason
        assert result.global_vector[0]*10 == pytest.approx(-8.1276, abs=1e-4)
        assert result.global_vector[1]*10 == pytest.approx(-7.9359, abs=1e-4)
        assert (result.global_vector[1]-result.global_vector[0])*10 == pytest.approx(0.19168571, abs=2e-6)


    def test_nonminimum_keeps_electronic_but_masks_frequency_and_entropy(store):
        result = store.get("B1(B2OCCO2)OCCO1", "boron")
        assert result.available
        assert result.global_vector[11:].tolist() == [0.0, 0.0, 0.0]
        assert result.global_vector[7:10].tolist() == [0.0, 0.0, 0.0]


    def test_unknown_molecule_masks_and_absent_root(store, tmp_path):
        missing = store.get("CCO", "sub")
        assert not missing.available
        assert not missing.atom.any() and not missing.global_vector.any()
        assert missing.edge_vector(0, 1) == [0.0]*6
        with pytest.raises(FileNotFoundError, match="manifest"):
            XTBFeatureStore(tmp_path)


    def test_cache_reload_and_invalidation(store):
        before = store.get(example_smiles(), "sub")
        new = XTBFeatureStore(store.root, store.cache_dir)
        np.testing.assert_array_equal(new.get(example_smiles(), "sub").atom, before.atom)
        old_count = len(list(store.cache_dir.glob("*.npz")))
        path = store.root / "calculations/sub/39ba84e13e6e6f35/wbo"
        path.write_text(path.read_text()+"\n")
        refreshed = XTBFeatureStore(store.root, store.cache_dir)
        np.testing.assert_array_equal(refreshed.get(example_smiles(), "sub").atom, before.atom)
        assert len(list(store.cache_dir.glob("*.npz"))) == old_count+1


    def test_corrupted_cache_is_rebuilt_from_raw_outputs(store, tmp_path):
        fresh = XTBFeatureStore(store.root, tmp_path)
        before = fresh.get(example_smiles(), "sub")
        assert before.available
        cache = next(tmp_path.glob("*.npz"))
        cache.write_bytes(b"PK\x03\x04corrupted archive")
        rebuilt = XTBFeatureStore(store.root, tmp_path).get(example_smiles(), "sub")
        assert rebuilt.available, rebuilt.reason
        np.testing.assert_array_equal(rebuilt.atom, before.atom)


    def test_default_and_explicit_store_share_coverage(monkeypatch, store):
        monkeypatch.setenv("MCGNN_XTB_ROOT", str(store.root))
        default = get_xtb_store()
        assert default is get_xtb_store(None) is get_xtb_store(root=store.root)
        default.get("CCO", "sol")
        assert {"role": "sol", "smiles": "CCO", "reason": "not calculated"} in default.report()["missing_molecules"]


    @pytest.fixture(scope="module")
    def graph_data():
        frame = pd.read_csv(ROOT / "data/Database Ir.csv")
        multi = frame[frame.bpin_index_sub.str.contains(",")].iloc[[0]]
        frame = pd.concat([frame.iloc[:3], multi], ignore_index=True)
        store = XTBFeatureStore()
        dataset, width = build_dataset(frame, filter_valid_indices(frame), torch.device("cpu"),
                                      fp_nbits=32, include_3d_descriptors=False, xtb_store=store,
                                      progress_interval=0)
        return frame, dataset, width, store


    def new_model(width):
        torch.manual_seed(123)
        return ReactionEnergyPredictionModel(width, environment_global_feature_dimension(include_3d=False),
                                            reactant_hidden_feats=16, reactant_out_feats=12,
                                            product_hidden_feats=16, product_out_feats=12,
                                            fc_hidden_feats=16, catalyst_context_dim=8, dropout=0.0)


    def metadata():
        return {"xtb_feature_config": xtb_feature_config(),
                "node_feature_schema": list(NODE_FEATURE_NAMES), "edge_feature_schema": list(EDGE_FEATURE_NAMES),
                "fingerprint_config": {"n_bits_per_fingerprint": 32},
                "global_descriptor_config": global_descriptor_config(include_3d=False),
                "model_config": {"dropout": 0.0}}


    def test_graph_offsets_multisite_and_virtual_metal_masks(graph_data):
        _, dataset, width, _ = graph_data
        assert len(dataset) == 4
        assert width == 32*2+17*2+3*14
        for _, env, reactant, product, temperature, _ in dataset:
            assert temperature > 273.15
            for graph in (env, reactant, product):
                assert graph.ndata['node'].shape[1] == 27
                assert graph.edata['edge'].shape[1] == 16
                assert torch.isfinite(graph.ndata['node']).all()
                assert torch.isfinite(graph.edata['edge']).all()
            metal = reactant.ndata['node'][:, NODE_FEATURE_NAMES.index('is_metal')].bool()
            assert int(metal.sum()) == 1
            assert reactant.ndata['node'][metal, 1].item() == 0.0  
            assert not reactant.ndata['node'][metal, 17:25].any()
            virtual = reactant.edata['edge'][:, -1].bool()
            assert virtual.any() and not reactant.edata['edge'][virtual, 10:15].any()
            assert reactant.ndata['node'][:, -1].sum() > 0  
        multi_graph = dataset[-1][2]
        sites = multi_graph.ndata['node'][:, 4].bool()
        assert sites.sum() == 2
        assert (multi_graph.ndata['node'][sites, NODE_FEATURE_NAMES.index('xtb_atom_available')] == 1).all()
        assert float(dataset[-1][-1]) == float(graph_data[0].iloc[-1]['G (kcal/mol)'])


    def test_checkpoint_and_prediction_equal_dataset_pipeline(graph_data, tmp_path):
        frame, dataset, width, _ = graph_data
        model = new_model(width).eval()
        _, env, rea, prod, temperature, _ = collate_regression(
            dataset, torch.device('cpu')
        )
        with torch.no_grad():
            delta_g, delta_h, delta_s = model(
                env, rea, prod, temperature, return_components=True
            )
            expected = delta_g.squeeze(-1).numpy()
            torch.testing.assert_close(
                delta_g,
                delta_h - temperature.reshape(-1, 1) * delta_s / 1000.0,
            )
        path = tmp_path/'model.pth'
        torch.save({'model_state_dict':model.state_dict(),'metadata':metadata()}, path)
        restored, _, config = load_regression_checkpoint(path, torch.device('cpu'))
        predicted, _ = predict_dataframe(restored, frame, torch.device('cpu'), batch_size=2)
        np.testing.assert_allclose(predicted['Predicted G (kcal/mol)'], expected, rtol=1e-5, atol=1e-5)
        assert predicted.status.eq('OK').all()
        assert config['catalyst_context_dim'] == 8
        broken = metadata()
        broken['xtb_feature_config']['schema_version'] = 'old'
        with pytest.raises(ValueError, match='metadata'):
            build_regression_model_from_checkpoint(model.state_dict(), broken)


    def test_context_retains_metal_identity_and_gets_gradients(graph_data):
        _, dataset, width, _ = graph_data
        model = new_model(width)
        _, env, rea, prod, temperature, target = collate_regression(
            dataset[:2], torch.device('cpu')
        )
        with rea.local_scope():
            zero_states = torch.zeros(rea.num_nodes(), 12)
            original = model.catalyst_context(rea, zero_states)
            changed = rea.ndata['node'].clone()
            mask = changed[:, NODE_FEATURE_NAMES.index('is_metal')].bool()
            changed[mask, 0] = 27.0
            rea.ndata['node'] = changed
            cobalt = model.catalyst_context(rea, zero_states)
            assert not torch.allclose(original, cobalt)
        loss = ((model(env, rea, prod, temperature).squeeze(-1)-target)**2).mean()
        loss.backward()
        assert model.catalyst_context.metal_projection[0].weight.grad.abs().sum() > 0
        assert model.reactant_gnn.gnn_layer.res_fc.weight.grad.abs().sum() > 0


    def test_transfer_freezes_encoder_weights_and_dropout(graph_data):
        _, dataset, width, _ = graph_data
        model = new_model(width)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(('catalyst_context.', 'fc1.', 'fc2.'))
        before = copy.deepcopy(model.environment_gnn.state_dict())
        head_before = model.fc2.weight.detach().clone()
        history = train_regressor(model, dataset[:2], dataset[2:], torch.device('cpu'),
                                  num_epochs=1, batch_size=2)
        assert np.isfinite(history['train_loss']).all()
        for name, value in model.environment_gnn.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert not torch.equal(head_before, model.fc2.weight)
        assert not model.environment_gnn.training
