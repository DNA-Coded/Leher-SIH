import sys
import os
import logging
from threading import Lock
from contextlib import contextmanager
from apps.api.leher.paths import REPO_ROOT

logger = logging.getLogger(__name__)

# To import run_rakshak, we need to add the parent of core to sys.path
_jal_chakra_dir = str(REPO_ROOT / "services" / "jal-chakra")
if _jal_chakra_dir not in sys.path:
    sys.path.insert(0, _jal_chakra_dir)

from core.rakshak import run_rakshak

_rakshak_lock = Lock()
_execution_state = {"status": "idle"}

def get_execution_status():
    return dict(_execution_state)

@contextmanager
def safe_cwd(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)

def execute_rakshak_job():
    # Attempt to acquire lock without blocking to prevent overlapping runs
    if not _rakshak_lock.acquire(blocking=False):
        logger.warning("Rakshak job skipped because another execution is active")
        return {"status": "skipped", "reason": "rakshak_run_already_in_progress"}

    logger.info("Rakshak job started")
    _execution_state["status"] = "running"
    try:
        with safe_cwd(REPO_ROOT):
            result = run_rakshak()
        logger.info("Rakshak job completed successfully")
        _execution_state["status"] = "idle"
        return {"status": "success", "data": result}
    except Exception as e:
        logger.exception("Rakshak job failed: %s", str(e))
        _execution_state["status"] = "failed"
        raise
    finally:
        _rakshak_lock.release()
