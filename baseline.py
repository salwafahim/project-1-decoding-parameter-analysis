"""Baseline run: temperature 0.7, top_k 40, top_p 0.9 (both models)."""
from common import BASELINE, run_group

run_group("baseline", [BASELINE])