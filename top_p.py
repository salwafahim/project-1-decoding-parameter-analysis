"""top_p sweep: only top_p changes, the other two stay at baseline (both models)."""
from common import run_group, sweep_settings

run_group("top_p", sweep_settings("top_p"))