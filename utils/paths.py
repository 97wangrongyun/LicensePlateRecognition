from pathlib import Path
import os
import yaml

# 项目根目录（以 utils/ 为基准）
BASE_DIR = Path(__file__).resolve().parents[1]

# 允许通过环境变量覆盖
DATA_ROOT = Path(os.getenv("DATA_ROOT", BASE_DIR / "data"))
RUNS_DIR  = Path(os.getenv("RUNS_DIR",  BASE_DIR / "runs"))
DEVICE    = os.getenv("DEVICE", "0")  # "0" or "cpu"
CFG_PATH  = Path(os.getenv("APP_CONFIG", BASE_DIR / "config" / "app.yaml"))

def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def abspath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (BASE_DIR / p)