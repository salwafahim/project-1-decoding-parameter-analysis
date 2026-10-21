"""top_k sweep: only top_k changes, the other two stay at baseline (both models)."""
from common import run_group, sweep_settings

run_group("top_k", sweep_settings("top_k"))