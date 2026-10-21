"""Shared engine used by baseline.py, temperature.py, top_k.py and top_p.py.

Every script runs BOTH models on its own group of settings and saves ONE csv:
    results/<group>_results.csv   (the 'model' column tells llama from qwen)

Stopping (Ctrl+C) is safe: run the same script again and finished runs are skipped.
Add --test to any script for a tiny trial run (saved in results_test/).
"""

import csv
import json
import sys
import time
from pathlib import Path

import ollama
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = BASE_DIR / "prompts" / "prompts.json"

MODELS = ["llama3.2:1b", "qwen2.5:1.5b"]
REPEATS = 5          # times each prompt is asked per setting
NUM_PREDICT = 200    # max tokens per answer (keeps runs fast)

# Baseline: the two parameters that are not being changed stay at these values.
BASELINE = {"temperature": 0.7, "top_k": 40, "top_p": 0.9}
# Values tested besides the baseline value.
SWEEPS = {
    "temperature": [0.2, 1.0, 1.5, 2.0],
    "top_k": [1, 10, 100],
    "top_p": [0.5, 0.7, 1.0],
}

FIELDS = [
    "model", "temperature", "top_k", "top_p", "prompt_id", "category",
    "repeat", "response", "response_time_sec", "error",
]


def sweep_settings(parameter: str) -> list[dict]:
    """Settings where only `parameter` changes and the other two stay at baseline."""
    settings = []
    for value in SWEEPS[parameter]:
        setting = dict(BASELINE)
        setting[parameter] = value
        settings.append(setting)
    return settings


def load_done(path: Path) -> set:
    """Keys of runs that already finished without an error."""
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["error"] == "":
                done.add((
                    row["model"], float(row["temperature"]), int(row["top_k"]),
                    float(row["top_p"]), int(row["prompt_id"]), int(row["repeat"]),
                ))
    return done


def ask_model(model: str, prompt: str, setting: dict, repeat: int) -> str:
    """One call to Ollama. seed = repeat: repeats differ, but a rerun gives the same results."""
    options = {
        "temperature": setting["temperature"],
        "top_k": setting["top_k"],
        "top_p": setting["top_p"],
        "num_predict": NUM_PREDICT,
        "seed": repeat,
    }
    reply = ollama.generate(model=model, prompt=prompt, options=options, keep_alive="30m")
    return reply["response"]


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def run_group(name: str, settings: list[dict]) -> None:
    """Run both models on all `settings` and save results/<name>_results.csv."""
    test_mode = "--test" in sys.argv
    results_dir = BASE_DIR / ("results_test" if test_mode else "results")
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{name}_results.csv"

    with open(PROMPTS_FILE, "r", encoding="utf-8") as file:
        prompts = json.load(file)
    repeats = REPEATS
    if test_mode:  # one prompt per category, one repeat, first setting only
        first_per_category = {}
        for item in prompts:
            first_per_category.setdefault(item["category"], item)
        prompts = list(first_per_category.values())
        settings = settings[:1]
        repeats = 1

    # Order: model -> setting -> prompt -> repeat (each model is loaded only once).
    tasks = [
        (model, setting, item, repeat)
        for model in MODELS
        for setting in settings
        for item in prompts
        for repeat in range(1, repeats + 1)
    ]
    done_keys = load_done(path)
    is_new = not path.exists()

    def task_key(task):
        model, setting, item, repeat = task
        return (
            model, float(setting["temperature"]), int(setting["top_k"]),
            float(setting["top_p"]), item["id"], repeat,
        )

    total = len(tasks)
    pending = [task for task in tasks if task_key(task) not in done_keys]

    print(f"Group: {name} | models: {MODELS}")
    print(f"Settings: {len(settings)} | prompts: {len(prompts)} | repeats: {repeats}")
    print(f"Total runs: {total} | already done: {total - len(pending)} | saving to: {path}\n")

    started = time.time()
    bar = tqdm(total=total, initial=total - len(pending), desc=name, unit="run", dynamic_ncols=True)
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        try:
            for model, setting, item, repeat in pending:
                response, error = "", ""
                begin = time.time()
                for _ in range(2):  # one retry if Ollama hiccups
                    try:
                        response = ask_model(model, item["prompt"], setting, repeat)
                        error = ""
                        break
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)
                        time.sleep(2)
                seconds = time.time() - begin

                writer.writerow({
                    "model": model,
                    "temperature": setting["temperature"],
                    "top_k": setting["top_k"],
                    "top_p": setting["top_p"],
                    "prompt_id": item["id"],
                    "category": item["category"],
                    "repeat": repeat,
                    "response": response,
                    "response_time_sec": round(seconds, 2),
                    "error": error,
                })
                file.flush()

                bar.update(1)
                bar.set_postfix_str(
                    f"{model} T={setting['temperature']} k={setting['top_k']} "
                    f"p={setting['top_p']} prompt {item['id']} rep {repeat}"
                )
                if error:
                    tqdm.write(f"ERROR on prompt {item['id']}: {error[:80]}")
        except KeyboardInterrupt:
            bar.close()
            print("\nStopped. Everything so far is saved. Run the same command to continue.")
            return
    bar.close()

    print(f"\nDone. Time this session: {format_time(time.time() - started)}")
    print(f"Saved: {path}")