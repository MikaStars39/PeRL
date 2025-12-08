import argparse
import os
import sys
import json
from typing import List
from enum import Enum
from flask import Flask, jsonify, render_template
from datetime import datetime

app = Flask(__name__)
RESULT_DIR = ""


class StatusType(Enum):
    RUNNING = "running"
    FINISHED = "finished"


class TaskResult:
    task_name: str
    rollout_n: int
    result_dir: str
    status: StatusType = StatusType.FINISHED
    size: int = 0
    avg: float = 0.0
    max: float = 0.0
    min: float = 0.0
    std: float = 0.0
    process: float = 1.0
    last_update: datetime = datetime.fromtimestamp(0)

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "rollout_n": self.rollout_n,
            "result_dir": self.result_dir,
            "status": self.status.value,
            "size": self.size,
            "avg": self.avg,
            "max": self.max,
            "min": self.min,
            "std": self.std,
            "process": self.process,
            "last_update": self.last_update.isoformat(),
        }

    def __init__(self, task_name: str, result_dir):
        if not os.path.exists(result_dir):
            raise FileNotFoundError(f"Result directory '{result_dir}' does not exist.")

        self.avg = 0.0
        self.max = 0.0
        self.min = 0.0
        self.std = 0.0
        self.size = 0
        self.process = 1.0
        self.status = StatusType.FINISHED
        self.last_update = datetime.fromtimestamp(0)

        if os.path.exists(os.path.join(result_dir, "process.txt")):
            self.status = StatusType.RUNNING
            with open(os.path.join(result_dir, "process.txt"), "r") as f:
                process_str = f.read()
            dot_count = process_str.count(".")
            x_count = process_str.count("X")
            if x_count + dot_count > 0:
                self.process = x_count / float(x_count + dot_count)
            else:
                self.process = 0.0
            self.size = x_count + dot_count
            self.rollout_n = process_str.strip().count("\n") + 1
            self.last_update = datetime.fromtimestamp(
                os.path.getmtime(os.path.join(result_dir, "process.txt"))
            )

        elif os.path.exists(os.path.join(result_dir, "result.json")):
            self.status = StatusType.FINISHED
            self.process = 1.0
            with open(os.path.join(result_dir, "result.json")) as f:
                result_json = json.load(f)
            self.rollout_n = int(result_json["rollout_n"])
            self.size = len(result_json["raw"]) * int(result_json["rollout_n"])
            self.avg = float(result_json["summary"]["avg"])
            self.max = float(result_json["summary"]["max"])
            self.min = float(result_json["summary"]["min"])
            self.std = float(result_json["summary"]["std"])
            self.last_update = datetime.fromtimestamp(
                os.path.getmtime(os.path.join(result_dir, "result.json"))
            )

        else:
            self.rollout_n = 0
            self.process = 0.0
            raise FileNotFoundError(f"Broken result directory: {result_dir}")

        self.task_name = f"{task_name}@{self.rollout_n}"
        self.result_dir = result_dir


class ExpResult:
    exp_name: str
    result_dir: str
    status: StatusType
    results: List[TaskResult]
    size: int
    avg: float
    max: float
    min: float
    std: float
    process: float
    last_update: datetime = datetime.fromtimestamp(0)

    def to_dict(self):
        return {
            "exp_name": self.exp_name,
            "result_dir": self.result_dir,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "size": self.size,
            "avg": self.avg,
            "max": self.max,
            "min": self.min,
            "std": self.std,
            "process": self.process,
            "last_update": self.last_update.isoformat(),
        }

    def add_task_result(self, task_result: TaskResult):
        self.results.append(task_result)

        full_size = self.size + task_result.size
        # Simple weighted average approximation (note: std dev aggregation is mathematically incorrect here but keeping style)
        if full_size > 0:
            self.avg = (
                self.avg * self.size + task_result.avg * task_result.size
            ) / full_size
            self.max = (
                self.max * self.size + task_result.max * task_result.size
            ) / full_size
            self.min = (
                self.min * self.size + task_result.min * task_result.size
            ) / full_size
            self.std = (
                self.std * self.size + task_result.std * task_result.size
            ) / full_size
            self.process = (
                self.process * self.size + task_result.process * task_result.size
            ) / full_size
            self.size = full_size

        self.last_update = max(self.last_update, task_result.last_update)

    def __init__(self, exp_name: str, result_dir: str):
        self.exp_name = exp_name
        self.result_dir = result_dir

        self.status = StatusType.FINISHED
        self.results = []
        self.size = 0
        self.avg = 0.0
        self.max = 0.0
        self.min = 0.0
        self.std = 0.0
        self.process = 1.0
        self.last_update = datetime.fromtimestamp(0)

        if not os.path.exists(result_dir):
            return

        for task_name in os.listdir(result_dir):
            if task_name == "logs" or task_name == "model":
                continue
            if os.path.isfile(os.path.join(result_dir, task_name)):
                continue

            # Directly load TaskResult, exceptions will propagate
            tr = TaskResult(task_name, os.path.join(result_dir, task_name))
            self.add_task_result(tr)

        if any(r.status == StatusType.RUNNING for r in self.results):
            self.status = StatusType.RUNNING


def get_results(result_dir: str) -> List[ExpResult]:
    result = []
    if not os.path.exists(result_dir):
        return result

    for exp_name in os.listdir(result_dir):
        if not os.path.isdir(os.path.join(result_dir, exp_name)):
            continue
        result.append(ExpResult(exp_name, os.path.join(result_dir, exp_name)))

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    results = get_results(RESULT_DIR)
    return jsonify([r.to_dict() for r in results])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor for evaluation results")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="The directory containing the evaluation results",
    )
    # Allow port configuration
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    # Adding serve argument to be consistent with previous read
    parser.add_argument(
        "--serve", action="store_true", help="Whether to serve the results"
    )
    args = parser.parse_args()

    RESULT_DIR = args.result_dir
    if not os.path.exists(RESULT_DIR):
        print(f"Error: Result directory '{RESULT_DIR}' does not exist.")
        sys.exit(-1)

    if args.serve:
        print(f"Viewer for evaluation results in {RESULT_DIR}")
        print(f"Serving at http://0.0.0.0:{args.port}")

        # Disable reloader to avoid issues in some environments, or keep it for dev
        app.run(host="0.0.0.0", port=args.port, debug=False)

    else:
        # Default behavior (print json) if not serving
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, datetime):
                    return o.isoformat()
                return super().default(o)

        print(
            json.dumps(
                [r.to_dict() for r in get_results(RESULT_DIR)],
                indent=2,
                ensure_ascii=False,
                cls=DateTimeEncoder,
            )
        )
