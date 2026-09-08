import pyarrow as pa
import polars as pl
import xarray as xr

def encode_dataset_to_arrow_ipc(ds: xr.Dataset, variable: str) -> bytes:
    """
    Converts a bounded 2D xarray dataset to Apache Arrow IPC format.
    Returns: latitude, longitude, and value fields preserving NaNs.
    """
    # xarray to pandas dataframe, then reset multi-index
    df = ds.to_dataframe().reset_index()
    
    # Filter the exact required columns
    required_cols = ["latitude", "longitude", variable]
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for Arrow conversion: {col}")
            
    out_df = pl.from_pandas(df[required_cols])
    out_df = out_df.rename({variable: "value"})
    
    table = out_df.to_arrow()
    
    # Write to IPC format
    sink = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)
        
    return sink.getvalue().to_pybytes()

def encode_vector_dataset_to_arrow_ipc(ds: xr.Dataset, variables: list[str]) -> bytes:
    """
    Converts a bounded 2D xarray dataset containing multiple variables to Arrow IPC format.
    Returns: latitude, longitude, and specified variable fields.
    """
    df = ds.to_dataframe().reset_index()
    
    required_cols = ["latitude", "longitude"] + variables
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for Arrow conversion: {col}")
            
    out_df = pl.from_pandas(df[required_cols])
    
    table = out_df.to_arrow()
    
    sink = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)
        
    return sink.getvalue().to_pybytes()

def encode_polars_to_arrow_ipc(df: pl.DataFrame) -> bytes:
    """
    Converts a Polars DataFrame directly to an Apache Arrow IPC stream.
    Used for hazards and other tabular ML outputs.
    """
    table = df.to_arrow()
    
    sink = pa.BufferOutputStream()
    with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)
        
    return sink.getvalue().to_pybytes()
