from pathlib import Path
from environs import Env

REPO_ROOT = f"{Path(__file__).parent.resolve()}/../.."
ASSETS_PATH = f"{REPO_ROOT}/assets"

env = Env(expand_vars=True)
env_file_path = Path(f"{Path.home()}/.config/flexavatar/.env")
if env_file_path.exists():
    env.read_env(str(env_file_path), recurse=False)

with env.prefixed("FLEXAVATAR_"):
    FLEXAVATAR_INPUTS_PATH = env("INPUTS_PATH", f"{REPO_ROOT}/data/inputs")
    FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH = env("PIXEL3DMM_PROCESSING_PATH", f"{REPO_ROOT}/data/pixel3dmm_processing")
    FLEXAVATAR_MODELS_PATH = env("MODELS_PATH", f"{REPO_ROOT}/models")