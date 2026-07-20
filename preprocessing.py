import os
import re
import pandas as pd
import numpy as np
import io
import configuration as cfg

def read_data_with_fallback(file_path, sheet_name):
    """
    Reads data with an iterative encoding fallback to prevent UnicodeDecodeError.
    Operates at the byte level to ensure compatibility across data modalities.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at {file_path}")

    with open(file_path, 'rb') as f:
        raw_bytes = f.read()

    for enc in encodings:
        try:
            raw_bytes.decode(enc, errors='strict')
            return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name, engine='openpyxl')
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            
    raise ValueError(f"Failed to decode and load {file_path} using fallbacks: {encodings}")

def clean_text(text):
    """
    Applies minimalist preprocessing aligned with IndoBERT (uncased) training state.
    Strictly handles np.nan and float mitigation.
    """
    if pd.isna(text):
        return ""
    
    try:
        text_str = str(text)
        text_str = text_str.lower()
        text_str = re.sub(r'\burl\b', ' ', text_str)
        text_str = re.sub(r'\s+', ' ', text_str).strip()
        return text_str
    
    except Exception:
        return ""

def process_and_cache_pipeline(file_path, sheet_name, text_col, label_col, cache_path):
    if os.path.exists(cache_path):
        print(f"[INFO] Cache located at {cache_path}. Validating schema...")
        try:
            df_cache = pd.read_excel(cache_path, engine='openpyxl')
            
            if label_col not in df_cache.columns:
                raise KeyError(f"Target column '{label_col}' missing in cached data.")
            
            print("[INFO] Cache validation successful. Bypassing extraction pipeline...")
            return df_cache
            
        except Exception as e:
            print(f"[WARNING] Cache structural shift detected: {e}")
            print("[WARNING] Bypassing cache. Re-initializing preprocessing pipeline...")
    
    print(f"[INFO] Extracting raw data from {file_path}...")
    df = read_data_with_fallback(file_path, sheet_name)
    
    df[text_col] = df[text_col].fillna("")
    
    print("[INFO] Applying minimalist mandatory cleansing...")
    df[text_col] = df[text_col].apply(clean_text)
    
    df = df[df[text_col] != ""]
    
    initial_shape = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    dedup_shape = len(df)
    print(f"[INFO] Global Deduplication: Removed {initial_shape - dedup_shape} duplicate records.")
    
    print(f"[INFO] Pipeline complete. Caching preprocessed state to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_excel(cache_path, index=False, engine='openpyxl')
    
    return df

def get_data():
    print("\n--- Initializing Input Data Pipeline ---")
    df_input = process_and_cache_pipeline(
        cfg.INPUT_DATA_FILE, 
        cfg.INPUT_DATA_SHEET, 
        cfg.TEXT_COLUMN_INPUT, 
        cfg.LABEL_COLUMN_INPUT, 
        cfg.PREPROCESSED_DATA_FILE
    )
    
    df_aug = None
    if getattr(cfg, 'USE_AUGMENTATION', False):
        print("\n--- Initializing Augmentation Data Pipeline ---")
        df_aug = process_and_cache_pipeline(
            cfg.MASTER_AUGMENTATION_FILE, 
            cfg.MASTER_AUGMENTATION_SHEET, 
            cfg.TEXT_COLUMN_AUGMENTATION, 
            cfg.LABEL_COLUMN_AUGMENTATION, 
            cfg.PREPROCESSED_AUGMENTATION_FILE
        )
        
    return df_input, df_aug

if __name__ == "__main__":
    get_data()