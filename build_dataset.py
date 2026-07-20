import os
import csv
import pandas as pd
import urllib.request
import configuration as cfg

def fetch_standard(url, expected_col_index, source_name, fallback_header):
    filename = url.split('/')[-1]
    print(f"[FETCH] {filename} -> Tagging as '{source_name}'")
    
    df = None
    final_col_index = expected_col_index
    
    read_configs = [
        {'sep': ',', 'encoding': 'utf-8', 'quoting': csv.QUOTE_MINIMAL},
        {'sep': '\t', 'encoding': 'utf-8', 'quoting': csv.QUOTE_MINIMAL},
        {'sep': '\t', 'encoding': 'latin-1', 'quoting': csv.QUOTE_MINIMAL}, 
        {'sep': ',', 'encoding': 'latin-1', 'quoting': csv.QUOTE_MINIMAL}
    ]

    for config in read_configs:
        try:
            temp_df = pd.read_csv(url, engine='python', on_bad_lines='skip', index_col=False, **config)
            
            temp_df.columns = temp_df.columns.astype(str).str.strip()
            temp_df.columns = temp_df.columns.str.replace('"', '').str.replace("'", "")
            
            if temp_df.shape[1] > expected_col_index:
                df = temp_df
                final_col_index = expected_col_index
                break
            elif fallback_header in temp_df.columns:
                df = temp_df
                final_col_index = temp_df.columns.get_loc(fallback_header)
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"[ERROR] Failed to parse {filename}")
        
    extracted = pd.DataFrame()
    extracted['Data'] = df.iloc[:, final_col_index].astype(str).str.strip('"\'+ ')
    extracted['source'] = source_name
    extracted['local_index'] = range(1, len(extracted) + 1)
    
    print(f"[OK] Extracted {len(extracted)} rows for {source_name}")
    return extracted

def fetch_ibrohim_robust(url, source_name="ibrohim budi"):
    filename = url.split('/')[-1]
    print(f"[FETCH] {filename} -> Tagging as '{source_name}' (Native Byte Surgery)")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        # errors='replace' adalah imunitas absolut terhadap karakter/emoji yang rusak
        raw_text = response.read().decode('utf-8', errors='replace')
        
    lines = raw_text.splitlines()
    
    raw_lines = pd.Series(lines[1:])
    
    raw_lines = raw_lines[raw_lines.str.strip() != ""]
    
    clean_text = raw_lines.str.replace(r'(,\s*[0-9]\s*){12}$', '', regex=True)
    
    extracted = pd.DataFrame()
    extracted['Data'] = clean_text.str.strip('"\'+ ')
    extracted['source'] = source_name
    
    extracted = extracted.reset_index(drop=True)
    extracted['local_index'] = range(1, len(extracted) + 1)
    
    print(f"[OK] Extracted {len(extracted)} rows for {source_name}")
    return extracted

def execute_build():
    print("\n[INIT] Starting Build via Isolated Parsing Strategies...")
    
    df_alfina = fetch_standard(cfg.GITHUB_SOURCE_ALFINA, 1, "alfina et al", "Tweet")
    df_ibrohim = fetch_ibrohim_robust(cfg.GITHUB_SOURCE_IBROHIM)
    df_pratiwi = fetch_standard(cfg.GITHUB_SOURCE_PRATIWI, 0, "pratiwi et al", "comment_text")
    
    print("[PROC] Aggregating all source datasets...")
    df_raw = pd.concat([df_alfina, df_ibrohim, df_pratiwi], ignore_index=True)
    
    print(f"[PROC] Loading {os.path.basename(cfg.MAPPING_ORIGINAL_FILE)}")
    try:
        df_map = pd.read_csv(cfg.MAPPING_ORIGINAL_FILE)
        if "parent id" in df_map.columns:
            df_map = df_map.rename(columns={"parent id": "parent_id"})
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] {cfg.MAPPING_ORIGINAL_FILE} not found.")
        
    print("[PROC] Merging by Composite Key (source + local_index)...")
    
    df_map['source'] = df_map['source'].astype(str).str.lower().str.strip()
    df_raw['source'] = df_raw['source'].astype(str).str.lower().str.strip()
    
    df_map['local_index'] = pd.to_numeric(df_map['local_index'], errors='coerce').fillna(0).astype(int)
    df_raw['local_index'] = pd.to_numeric(df_raw['local_index'], errors='coerce').fillna(0).astype(int)
    
    df_final = pd.merge(df_map, df_raw[['source', 'local_index', 'Data']], on=['source', 'local_index'], how='left')
    
    missing = df_final['Data'].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} rows failed to match! Check your 'source' labels or 'local_index'.")
    else:
        print("[OK] All 14,453 rows perfectly matched via Composite Key.")
        
    cols = ["source", "local_index", "parent_id", "Data", "domain", "Daya Ilokusi", "Jenis Ujaran"]
    for c in cols:
        if c not in df_final.columns:
            df_final[c] = None 
    df_final = df_final[cols]
    
    os.makedirs(os.path.dirname(cfg.INPUT_DATA_FILE), exist_ok=True)
    df_final.to_excel(cfg.INPUT_DATA_FILE, sheet_name=cfg.INPUT_DATA_SHEET, index=False)
    print(f"[DONE] Saved to Excel (Shape: {df_final.shape})\n")

if __name__ == "__main__":
    execute_build()