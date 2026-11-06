import csv
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem


XTB_SCHEMA_VERSION = "xtb_metal_v1"
XTB_PARSER_VERSION = 2
XTB_NODE_FEATURE_NAMES = (
    "xtb_partial_charge", "xtb_covcn_div_4", "xtb_alpha_div_20",
    "xtb_attached_h_charge_mean", "xtb_xh_wbo_mean", "xtb_xh_length_angstrom",
    "xtb_atom_available", "xtb_xh_available",
)
XTB_EDGE_FEATURE_NAMES = (
    "xtb_wiberg_bond_order", "xtb_bond_length_angstrom",
    "xtb_endpoint_charge_difference", "xtb_wbo_available",
    "xtb_geometry_available", "is_virtual_metal_edge",
)
XTB_GLOBAL_FEATURE_NAMES = (
    "xtb_homo_ev_div_10", "xtb_lumo_ev_div_10", "xtb_dipole_debye_div_10",
    "xtb_molecular_alpha_div_100", "xtb_charge_std", "xtb_charge_min",
    "xtb_charge_max", "xtb_min_positive_frequency_div_100",
    "xtb_fraction_modes_below_100", "xtb_entropy_298_cal_mol_k_div_100",
    "xtb_electronic_available", "xtb_minimum_geometry",
    "xtb_frequency_available", "xtb_thermo_available",
)
XTB_GLOBAL_DIM = len(XTB_GLOBAL_FEATURE_NAMES)
XTB_REACTION_GLOBAL_NAMES = tuple(
    f"{role}_{name}"
    for role in ("ligand", "molecule", "boron")
    for name in XTB_GLOBAL_FEATURE_NAMES
)
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
_ATOM_ROW = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+([A-Za-z]+)\s+({_NUMBER})\s+({_NUMBER})"
    rf"\s+({_NUMBER})\s+({_NUMBER})\s*$", re.M
)


def xtb_feature_config(enabled=True):
    return {
        "schema_version": XTB_SCHEMA_VERSION,
        "parser_version": XTB_PARSER_VERSION,
        "enabled": bool(enabled),
        "node_names": list(XTB_NODE_FEATURE_NAMES),
        "edge_names": list(XTB_EDGE_FEATURE_NAMES),
        "molecular_names": list(XTB_GLOBAL_FEATURE_NAMES),
        "reaction_global_names": list(XTB_REACTION_GLOBAL_NAMES),
        "method": "GFN2-xTB independent molecules; no computed metal complex",
        "selection": "same role and canonical structure; zero imaginary modes first, then lowest energy, then job_id",
        "missing_policy": "zero plus explicit availability masks; never drop reaction labels",
        "scaling": "fixed units encoded in feature names; no dataset-fitted preprocessing",
        "frontier_orbitals": "sp.out HOMO/LUMO indices with full-precision JSON energies; verify printed gap",
        "thermo_temperature_kelvin": 298.15,
        "imaginary_frequency_cutoff_cm1": -20.0,
        "metal_context": "raw metal + residual node state + candidate ligand N/P + marked sites",
    }


def validate_xtb_config(config):
    expected = xtb_feature_config()
    if not isinstance((config or {}).get("enabled"), bool):
        raise ValueError("Checkpoint must explicitly specify whether xTB features are enabled.")
    for key in ("schema_version", "parser_version", "node_names", "edge_names", "molecular_names",
                "reaction_global_names", "selection", "scaling", "thermo_temperature_kelvin",
                "imaginary_frequency_cutoff_cm1", "frontier_orbitals"):
        if (config or {}).get(key) != expected[key]:
            raise ValueError(
                f"Incompatible xTB metadata ({key}). Retrain using the current "
                "xTB/metal feature schema; legacy checkpoints cannot be reused."
            )


@dataclass
class XTBMolecule:
    atom: np.ndarray
    positions: np.ndarray
    wbo: dict
    global_vector: np.ndarray
    job_id: str = ""
    reason: str = ""

    @property
    def available(self):
        return bool(self.global_vector[10])

    def edge_vector(self, begin, end):
        value = np.zeros(len(XTB_EDGE_FEATURE_NAMES), dtype=np.float32)
        if not self.available:
            return value.tolist()
        distance = float(np.linalg.norm(self.positions[begin] - self.positions[end]))
        if not 0.4 < distance < 3.0:
            return value.tolist()
        value[1] = distance
        value[2] = abs(float(self.atom[begin, 0] - self.atom[end, 0]))
        value[4] = 1.0
        pair = tuple(sorted((begin, end)))
        if pair in self.wbo:
            value[0], value[3] = self.wbo[pair], 1.0
        return value.tolist()


def missing_xtb(mol, reason="not calculated"):
    return XTBMolecule(
        np.zeros((mol.GetNumAtoms(), len(XTB_NODE_FEATURE_NAMES)), dtype=np.float32),
        np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32), {},
        np.zeros(XTB_GLOBAL_DIM, dtype=np.float32), reason=reason,
    )


class XTBFeatureStore:


    def __init__(self, root=None, cache_dir=None):
        project = Path(__file__).resolve().parents[1]
        self.root = Path(root or os.environ.get("MCGNN_XTB_ROOT") or project / "data" / "xtb_ir_features").resolve()
        self.cache_dir = Path(cache_dir or project / "outputs" / "07_runtime_cache/xtb_cache" / XTB_SCHEMA_VERSION)
        manifest = self.root / "xtb_unique_molecule_features.csv"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Missing xTB manifest: {manifest}. Set MCGNN_XTB_ROOT to the "
                "feature directory, or explicitly disable xTB for an ablation."
            )
        self.manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.jobs = {}
        self.errors = {}
        self.requests = {}
        self._queries = {}
        self._raw = {}
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["role"] not in {"sub", "prod", "lig", "boron", "sol"}:
                    continue
                if row["status"] != "ok" or not re.fullmatch(r"[a-f0-9]{16}", row["job_id"]):
                    continue
                mol = Chem.MolFromSmiles(row["smiles"])
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                key = (row["role"], canonical)
                rank = (float(row.get("xtb_n_imag_freq") or "inf") != 0.0,
                        float(row["xtb_total_energy_Eh"]), row["job_id"])
                if key not in self.jobs or rank < self.jobs[key][0]:
                    self.jobs[key] = (rank, row)

    def get(self, smiles, role):
        key = (role, str(smiles))
        if key in self._queries:
            return self._queries[key]
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid {role} SMILES for xTB lookup: {smiles}")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        match = self.jobs.get((role, canonical))
        if match is None:
            result = missing_xtb(mol)
        else:
            row = match[1]
            try:
                raw = self._load_job(row)
                source = Chem.MolFromSmiles(row["smiles"])
                mapping = (tuple(range(mol.GetNumAtoms())) if str(smiles) == row["smiles"]
                           else source.GetSubstructMatch(mol, useChirality=True))
                if (len(mapping) != mol.GetNumAtoms() or source.GetNumAtoms() != mol.GetNumAtoms()
                        or source.GetNumBonds() != mol.GetNumBonds()):
                    raise ValueError("Cannot establish a complete stereochemical atom mapping")
                reverse = {old: new for new, old in enumerate(mapping)}
                wbo = {tuple(sorted((reverse[a], reverse[b]))): value
                       for (a, b), value in raw.wbo.items() if a in reverse and b in reverse}
                result = XTBMolecule(raw.atom[list(mapping)].copy(), raw.positions[list(mapping)].copy(),
                                     wbo, raw.global_vector.copy(), row["job_id"])
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                self.errors[row["job_id"]] = str(error)
                result = missing_xtb(mol, reason=str(error))
        self._queries[key] = result
        self.requests[key] = result.available
        return result

    def _load_job(self, row):
        job = self.root / "calculations" / row["role"] / row["job_id"]
        names = ("result.json", "xtbout.json", "xtbopt.xyz", "wbo", "sp.out", "vibspectrum")
        signature = [(name, (job / name).stat().st_size, (job / name).stat().st_mtime_ns)
                     for name in names]
        token = hashlib.sha256(json.dumps(
            [XTB_SCHEMA_VERSION, XTB_PARSER_VERSION, str(job), row["smiles"], signature]
        ).encode()).hexdigest()[:20]
        if token in self._raw:
            return self._raw[token]
        cache = self.cache_dir / f"{row['job_id']}_{token}.npz"
        result = None
        if cache.is_file():
            try:
                with np.load(cache, allow_pickle=False) as arrays:
                    result = XTBMolecule(arrays["atom"], arrays["positions"],
                        {(int(a), int(b)): float(v) for a, b, v in arrays["wbo"]},
                        arrays["global_vector"], row["job_id"])
                if (result.atom.shape != (Chem.MolFromSmiles(row["smiles"]).GetNumAtoms(), 8)
                        or result.positions.shape != (result.atom.shape[0], 3)
                        or result.global_vector.shape != (XTB_GLOBAL_DIM,)
                        or any(not np.isfinite(v) or not (0 <= a < len(result.atom) and 0 <= b < len(result.atom))
                               for (a, b), v in result.wbo.items())
                        or not all(np.isfinite(a).all() for a in
                                   (result.atom, result.positions, result.global_vector))):
                    result = None
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                result = None
        if result is None:
            result = self._parse_job(job, row)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, atom=result.atom, positions=result.positions,
                                        global_vector=result.global_vector,
                                        wbo=np.asarray([(a, b, v) for (a, b), v in result.wbo.items()],
                                                       dtype=float).reshape(-1, 3))
                os.replace(temporary, cache)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        self._raw[token] = result
        return result

    def _parse_job(self, job, row):
        data = json.loads((job / "xtbout.json").read_text(encoding="utf-8-sig"))
        result = json.loads((job / "result.json").read_text(encoding="utf-8-sig"))
        if data.get("method") != "GFN2-xTB" or result.get("status") != "ok":
            raise ValueError("Unsupported xTB method or unsuccessful calculation")
        mol = Chem.MolFromSmiles(row["smiles"])
        if result.get("xtb_charge") != Chem.GetFormalCharge(mol):
            raise ValueError("xTB total charge does not match the molecule")
        lines = (job / "xtbopt.xyz").read_text().splitlines()
        n = int(lines[0])
        xyz = [line.split() for line in lines[2:2+n]]
        symbols = [line[0] for line in xyz]
        count = mol.GetNumAtoms()
        if symbols[:count] != [a.GetSymbol() for a in mol.GetAtoms()]:
            raise ValueError("Original SMILES/XYZ atom sequence mismatch")
        positions = np.asarray([[float(x) for x in line[1:4]] for line in xyz])
        charges = np.asarray(data["partial charges"], dtype=float)
        if positions.shape != (n, 3) or charges.shape != (n,):
            raise ValueError("Invalid XYZ/charge dimensions")
        if not np.isfinite(positions).all() or not np.isfinite(charges).all():
            raise ValueError("Non-finite xTB coordinates or charges")
        for bond in mol.GetBonds():
            length = np.linalg.norm(positions[bond.GetBeginAtomIdx()] - positions[bond.GetEndAtomIdx()])
            if not 0.4 < length < 3.0:
                raise ValueError("Optimized geometry no longer matches a source chemical bond")
        atom = np.zeros((count, len(XTB_NODE_FEATURE_NAMES)), dtype=np.float32)
        atom[:, 0] = charges[:count]
        sp = (job / "sp.out").read_text(encoding="utf-8", errors="replace")
        atomic_rows = {int(values[0])-1: values for values in _ATOM_ROW.findall(sp)}
        if any(i not in atomic_rows for i in range(count)):
            raise ValueError("Missing atomic covCN/polarizability printout")
        for i in range(count):
            atom[i, 1] = float(atomic_rows[i][3]) / 4.0
            atom[i, 2] = float(atomic_rows[i][6]) / 20.0
        atom[:, 6] = 1.0
        all_wbo = {}
        for line in (job / "wbo").read_text().splitlines():
            values = line.split()
            if len(values) == 3:
                a, b, value = int(values[0])-1, int(values[1])-1, float(values[2])
                if not (0 <= a < n and 0 <= b < n and np.isfinite(value)):
                    raise ValueError("Invalid WBO atom pair")
                all_wbo[tuple(sorted((a, b)))] = value


        attached = {i: [] for i in range(count)}
        for h in range(count, n):
            if symbols[h] != "H":
                raise ValueError("Unexpected non-H atom after SMILES atom block")
            candidates = [(value, a if b == h else b)
                          for (a, b), value in all_wbo.items()
                          if h in (a, b) and (a if b == h else b) < count and value > 0.3]
            if candidates:
                value, heavy = max(candidates)
                length = float(np.linalg.norm(positions[heavy] - positions[h]))
                if 0.4 < length < 2.2:
                    attached[heavy].append((charges[h], value, length))
        for i, entries in attached.items():
            expected = mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)
            if entries and len(entries) == expected:
                atom[i, 3:6] = np.asarray(entries).mean(axis=0)
                atom[i, 7] = 1.0
        energies = np.asarray(data["orbital energies / eV"], dtype=float)
        frontier_indices = []
        for label in ("HOMO", "LUMO"):
            matches = re.findall(r"^\s*(\d+)[^\r\n]*\("+label+r"\)", sp, re.M)
            if not matches or not 0 <= int(matches[-1])-1 < len(energies):
                raise ValueError(f"Missing or invalid xTB-labeled {label} index")
            frontier_indices.append(int(matches[-1])-1)


        homo, lumo = energies[frontier_indices]
        if lumo < homo:
            raise ValueError("Unsupported frontier-orbital occupation ordering")
        if not np.isclose(lumo-homo, data["HOMO-LUMO gap / eV"], atol=1e-5, rtol=0):
            raise ValueError("Frontier-orbital energies disagree with the xTB gap")
        alpha = re.search(r"Mol\. α\(0\)[^\n:]*:\s*("+_NUMBER+")", sp)
        if alpha is None:
            raise ValueError("Missing molecular polarizability")
        glob = np.zeros(XTB_GLOBAL_DIM, dtype=np.float32)
        glob[:7] = [homo/10, lumo/10, np.linalg.norm(data["dipole / a.u."])*2.541746473/10,
                    float(alpha.group(1))/100, charges.std(), charges.min(), charges.max()]
        glob[10] = 1.0
        frequencies = []
        for line in (job / "vibspectrum").read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = line.split()
            if tokens and tokens[0].isdigit():
                numbers = [float(v) for v in tokens[1:] if re.fullmatch(_NUMBER, v)]
                if len(numbers) >= 2:
                    frequencies.append(numbers[0])
        minimum = (result.get("xtb_n_imag_freq") == 0 and
                   bool(frequencies) and min(frequencies) >= -20)
        glob[11] = float(minimum)
        positive = np.asarray([v for v in frequencies if v > 1.0])
        if minimum and len(positive):
            glob[7:9] = [positive.min()/100, np.mean(positive < 100)]
            glob[12] = 1.0
        entropy = result.get("thermo", {}).get("298.15", {}).get("S_calmolK")
        if minimum and entropy is not None and np.isfinite(entropy):
            glob[9], glob[13] = float(entropy)/100, 1.0
        if not np.isfinite(atom).all() or not np.isfinite(glob).all():
            raise ValueError("Non-finite xTB features")
        return XTBMolecule(atom, positions[:count].astype(np.float32),
                           {(a, b): v for (a, b), v in all_wbo.items() if a < count and b < count},
                           glob, row["job_id"])

    def report(self):
        coverage = {}
        for role in sorted({key[0] for key in self.requests}):
            values = [v for (r, _), v in self.requests.items() if r == role]
            coverage[role] = {"unique_smiles_requested": len(values), "available": sum(values),
                              "missing": len(values)-sum(values)}
        return {"schema_version": XTB_SCHEMA_VERSION, "parser_version": XTB_PARSER_VERSION,
                "root": str(self.root),
                "manifest_sha256": self.manifest_sha256, "coverage": coverage,
                "missing_molecules": [
                    {"role": role, "smiles": smiles, "reason": self._queries[(role, smiles)].reason}
                    for role, smiles in sorted(self.requests) if not self.requests[(role, smiles)]],
                "parse_errors": dict(self.errors)}


@lru_cache(maxsize=4)
def _cached_xtb_store(root):
    return XTBFeatureStore(root=root)


def get_xtb_store(root=None):


    project = Path(__file__).resolve().parents[1]
    root = Path(root or os.environ.get("MCGNN_XTB_ROOT") or project / "data" / "xtb_ir_features")
    return _cached_xtb_store(str(root.resolve()))


def reaction_xtb_vector(store, ligand_smiles, molecule_smiles, boron_smiles, molecule_role):
    if store is None:
        return np.zeros(len(XTB_REACTION_GLOBAL_NAMES), dtype=np.float32)
    return np.concatenate([store.get(ligand_smiles, "lig").global_vector,
                           store.get(molecule_smiles, molecule_role).global_vector,
                           store.get(boron_smiles, "boron").global_vector]).astype(np.float32)
