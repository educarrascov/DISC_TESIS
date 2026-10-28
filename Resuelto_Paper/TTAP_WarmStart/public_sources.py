"""Read immutable files already published in the user-selected repository.

Only GET requests are made. This module never publishes inputs or results.
"""

import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from objective_profiles import ObjectiveSpec


BASE_COMMIT = "07dd90042822e2cab84209fe81c9f31ecfe230a8"
ROOT = Path(__file__).resolve().parent
GPS_PATH = "20260829_TTAP_ambosMetodos_GPS_Impacto_Estandarizado.ipynb"
GPS_BLOB = "eac2e8b11e3436a2c0e378256a8db5cb6cfc5578"
HISTORICAL_CSV = {
    "A": ("ttap_experiment_A_4x4_impact_anytime_traced/metrics_A4x4_paper_20260808_112401-0400_matched_v3_impact_anytime.csv",
          "e16d63f59ebda826aa0481db2ac9d87e5cb26107", 600.),
    "B": ("ttap_experiment_B_6x7_stress_impact_anytime_traced/metrics_B6x7_stress_paper_20260825_195505-0400_matched_v3_impact_anytime_old6x7_operational_config.csv",
          "385663ea1b0228450cb93306d37e10e022f0a2b0", 1800.)}


def git_blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


def public_bytes(relative_path, expected_blob):
    if ".." in Path(relative_path).parts:
        raise ValueError("Unexpected source path")
    local = ROOT.parent / relative_path
    if local.is_file() and git_blob_sha(local.read_bytes()) == expected_blob:
        return local.read_bytes()
    url = ("https://raw.githubusercontent.com/educarrascov/DISC_TESIS/" + BASE_COMMIT
           + "/Resuelto_Paper/" + quote(relative_path, safe="/"))
    with urlopen(url, timeout=90) as response:
        content = response.read()
    if git_blob_sha(content) != expected_blob:
        raise ValueError("Published source differs from the declared base commit")
    return content


def historical_references(spec):
    if not spec.historical_compatible:
        raise ValueError("Historical references require the unchanged primary objective")
    from figure5_and_analysis import prepare, references
    frames = []
    for config, (path, blob, _) in HISTORICAL_CSV.items():
        frames.append(prepare(pd.read_csv(BytesIO(public_bytes(path, blob))), config))
    raw = pd.concat(frames, ignore_index=True)
    result = references(raw)
    seeds = raw[["Configuration", "Scenario", "Tasks", "Seed"]].drop_duplicates()
    result = result.merge(seeds, on=["Configuration", "Scenario", "Tasks"], validate="one_to_one")
    result["Objective SHA256"] = spec.fingerprint
    return result


def gps_notebook():
    return json.loads(public_bytes(GPS_PATH, GPS_BLOB))
