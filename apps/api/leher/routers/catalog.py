from fastapi import APIRouter, HTTPException, Request
from typing import List, Dict, Any
from apps.api.leher.catalog import service

router = APIRouter(tags=["Catalog"])

@router.get("/datasets", response_model=List[Dict[str, Any]])
def list_datasets(request: Request):
    """List all active datasets with their variables, depth ranges, time ranges."""
    db = getattr(request.app.state, "catalog_db", None)
    if not db:
        raise HTTPException(status_code=503, detail="Catalog database not initialized")
        
    datasets = service.get_active_datasets(db)
    return datasets

@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, request: Request):
    """Full metadata for a single dataset version."""
    db = getattr(request.app.state, "catalog_db", None)
    if not db:
        raise HTTPException(status_code=503, detail="Catalog database not initialized")
        
    dataset = service.get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset

@router.get("/datasets/{dataset_id}/variables")
def get_dataset_variables(dataset_id: str, request: Request):
    """List variables available in this dataset."""
    db = getattr(request.app.state, "catalog_db", None)
    if not db:
        raise HTTPException(status_code=503, detail="Catalog database not initialized")
        
    variables = service.get_dataset_variables(db, dataset_id)
    if variables is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return variables

@router.get("/variables/{var_id}/depths")
def get_variable_depths(var_id: str, request: Request):
    """List all native depth levels for a variable."""
    db = getattr(request.app.state, "catalog_db", None)
    if not db:
        raise HTTPException(status_code=503, detail="Catalog database not initialized")
        
    dataset = service.get_variable_active_dataset(db, var_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Variable '{var_id}' not found in any active dataset")
        
    depths = dataset.get("native_depths")
    if depths is None:
        return []
    return depths

@router.get("/variables/{var_id}/times")
def get_variable_times(var_id: str, request: Request):
    """List all available timesteps for a variable."""
    db = getattr(request.app.state, "catalog_db", None)
    if not db:
        raise HTTPException(status_code=503, detail="Catalog database not initialized")
        
    times = service.get_variable_times(db, var_id)
    if times is None:
        raise HTTPException(status_code=404, detail=f"Variable '{var_id}' not found or could not read times")
    return times
