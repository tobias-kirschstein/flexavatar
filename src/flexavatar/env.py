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
    FLEXAVATAR_AVATAR_CODE_PATH = env("AVATAR_CODE_PATH", f"{REPO_ROOT}/data/avatar_codes")
    FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH = env("PIXEL3DMM_PROCESSING_PATH", f"{REPO_ROOT}/data/pixel3dmm_processing")
    FLEXAVATAR_MATANYONE_PROCESSING_PATH = env("MATANYONE_PROCESSING_PATH", f"{REPO_ROOT}/data/matanyone_processing")
    FLEXAVATAR_DATASETS_PATH = env("DATASETS_PATH", f"{REPO_ROOT}/data/datasets")
    FLEXAVATAR_MODELS_PATH = env("MODELS_PATH", f"{REPO_ROOT}/models")
    FLEXAVATAR_RENDERINGS_PATH = env("RENDERINGS_PATH", f"{REPO_ROOT}/renderings")
    FLEXAVATAR_EVALUATIONS_PATH = env("EVALUATIONS_PATH", f"{REPO_ROOT}/evaluations")