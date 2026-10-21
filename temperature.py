"""temperature sweep: only temperature changes, the other two stay at baseline (both models)."""
from common import run_group, sweep_settings

run_group("temperature", sweep_settings("temperature"))