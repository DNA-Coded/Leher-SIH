import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from apps.api.leher.main import app, scheduler
from apps.api.leher.jobs.rakshak_job import execute_rakshak_job, _rakshak_lock
from apps.api.leher.paths import REPO_ROOT

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_lock():
    # Ensure lock is released before each test
    if _rakshak_lock.locked():
        _rakshak_lock.release()
    yield
    if _rakshak_lock.locked():
        _rakshak_lock.release()

@patch('apps.api.leher.jobs.rakshak_job.run_rakshak')
def test_job_working_directory_and_success(mock_run):
    mock_run.return_value = {"status": "success", "mocked": True}
    
    original_cwd = os.getcwd()
    
    # We can check what CWD was inside run_rakshak by using a side_effect
    cwd_inside = None
    def side_effect():
        nonlocal cwd_inside
        cwd_inside = os.getcwd()
        return {"status": "success", "mocked": True}
        
    mock_run.side_effect = side_effect
    
    result = execute_rakshak_job()
    
    assert cwd_inside == str(REPO_ROOT)
    assert os.getcwd() == original_cwd
    assert result == {"status": "success", "data": {"status": "success", "mocked": True}}
    assert not _rakshak_lock.locked()

@patch('apps.api.leher.jobs.rakshak_job.run_rakshak')
def test_job_working_directory_restored_on_exception(mock_run):
    mock_run.side_effect = ValueError("Mocked ML error")
    
    original_cwd = os.getcwd()
    
    with pytest.raises(ValueError, match="Mocked ML error"):
        execute_rakshak_job()
        
    assert os.getcwd() == original_cwd
    assert not _rakshak_lock.locked()

@patch('apps.api.leher.jobs.rakshak_job.run_rakshak')
def test_concurrent_execution_prevented(mock_run):
    # Lock it manually to simulate another run
    _rakshak_lock.acquire()
    
    result = execute_rakshak_job()
    
    assert result == {"status": "skipped", "reason": "rakshak_run_already_in_progress"}
    assert not mock_run.called

@patch('apps.api.leher.routers.ml._rakshak_lock')
@patch('apps.api.leher.jobs.rakshak_job.run_rakshak')
def test_manual_post_endpoint(mock_run, mock_lock):
    mock_lock.locked.return_value = False
    mock_run.return_value = {"status": "success", "mocked": True}
    
    # Use the test client context manager to trigger lifespan events
    with TestClient(app) as client:
        response = client.post("/api/v1/ml/run")
        assert response.status_code == 200
        assert response.json() == {"status": "accepted"}

def test_scheduler_configuration():
    # Start the app to trigger lifespan
    with TestClient(app):
        # Verify there is exactly one job named 'rakshak_hourly'
        jobs = scheduler.get_jobs()
        rakshak_jobs = [j for j in jobs if j.id == "rakshak_hourly"]
        
        assert len(rakshak_jobs) == 1
        job = rakshak_jobs[0]
        assert job.trigger.__class__.__name__ == "IntervalTrigger"
    
