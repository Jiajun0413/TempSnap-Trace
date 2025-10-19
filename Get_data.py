# Author: Liu Jiajun
# Date: 2025-10-19
#
# This script is written by Liu Jiajun. The corresponding article is "TempSnap-Trace: A Temporal Snapshot-Based Framework for Haplotype Network Tracing."
import os
import argparse
import subprocess
import re
import pandas as pd
import glob
import datetime
import logging
from typing import Optional, Tuple, List, Dict

# ------------------------ Common utility helpers (to reduce duplication) ------------------------ #

def normalize_dates(series: pd.Series) -> pd.Series:
    """Vectorized normalize of date strings supporting YYYY-MM-DD / YYYY-MM / YYYY.
    Empty/invalid -> NaT."""
    if series.empty:
        return pd.to_datetime(pd.Series([], dtype=str), errors='coerce')
    s = series.astype(str).str.strip().replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    parsed = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    mask_mm = parsed.isna() & s.str.match(r'^\d{4}-\d{2}$', na=False)
    if mask_mm.any():
        parsed.loc[mask_mm] = pd.to_datetime(s[mask_mm] + '-01', format='%Y-%m-%d', errors='coerce')
    mask_y = parsed.isna() & s.str.match(r'^\d{4}$', na=False)
    if mask_y.any():
        parsed.loc[mask_y] = pd.to_datetime(s[mask_y] + '-01-01', format='%Y-%m-%d', errors='coerce')
    return parsed

def compute_date_range(dates: pd.Series) -> Optional[str]:
    """Return YYYY_MM_DD-YYYY_MM_DD range string if possible."""
    try:
        valid = dates.dropna()
        if valid.empty: return None
        return f"{valid.min().strftime('%Y_%m_%d')}-{valid.max().strftime('%Y_%m_%d')}"
    except Exception:
        return None

def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def extract_mutation_ids(mutations_df: pd.DataFrame) -> pd.DataFrame:
    """Extract accession IDs from Query_ID pattern (split by |). Returns df with columns: raw_id, Accession, Mutations."""
    if mutations_df.empty: return pd.DataFrame(columns=['raw_id','Accession','Mutations'])
    q = mutations_df['Query_ID'].astype(str)
    parts = q.str.split('|')
    # Heuristic: if second field starts with EPI or looks like accession use it else first token before space.
    acc = parts.str[1].where(parts.str.len() >= 2, None)
    acc = acc.where(acc.str.match(r'^EPI[_A-Z]*_?ISL', na=False), None)
    # Fallback
    fallback = parts.str[0].str.split().str[0]
    acc = acc.fillna(fallback)
    muts_col = mutations_df['Mutations'].astype(str) if 'Mutations' in mutations_df.columns else pd.Series(['']*len(mutations_df))
    return pd.DataFrame({'raw_id': q, 'Accession': acc, 'Mutations': muts_col})


# ------------------------ FASTA parsing utilities ------------------------ #

def simple_fasta_parser(fasta_filename):
    """Simple FASTA parser (no Biopython). Yields (id, full_header, sequence)."""
    seq = []
    header = None
    seq_id = None
    try:
        with open(fasta_filename, 'r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        yield seq_id, header, ''.join(seq)
                    header = line[1:]
                    parts = header.split(None, 1)
                    seq_id = parts[0]
                    seq = []
                else:
                    if header is not None:
                        seq.append(line)
            if header is not None:
                yield seq_id, header, ''.join(seq)
    except FileNotFoundError:
        logging.error(f"FASTA file not found at {fasta_filename}")
    except Exception as e:
        logging.error(f"Error parsing FASTA file {fasta_filename}: {e}")


EMBED_SPLIT_PATTERN = re.compile(r"\|")

def parse_header_embedded_metadata(header: str) -> Dict[str, str]:
    """Parse embedded header: accession|date|location|lineage

    In embedded-only scenario we DO NOT have separate Clade field; set Clade='Unknown'.
    Example: >OR030923.1 |2022-06-29|Belgium|B.1
    Spaces around pipes tolerated.
    Returns dict with ID, Date, Location, Lineage, Clade.
    """
    if not header:
        return {"ID": "", "Date": "", "Location": "Unknown", "Lineage": "Unknown", "Clade": "Unknown"}
    parts = [p.strip() for p in EMBED_SPLIT_PATTERN.split(header)]
    id_part = parts[0].split()[0] if parts else ''
    date = parts[1] if len(parts) > 1 else ''
    location = parts[2] if len(parts) > 2 else 'Unknown'
    lineage = parts[3] if len(parts) > 3 else 'Unknown'
    if date and not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        # Keep raw; later normalization handles partial formats.
        pass
    return {"ID": id_part, "Date": date, "Location": location or 'Unknown', "Lineage": lineage or 'Unknown', "Clade": 'Unknown'}

def extract_date_from_header(header):
    """
    从序列标题中提取日期(YYYY-MM-DD格式)
    例如：>A/AICHI/1121/2009|EPI_ISL_66751|A_/_H1N1|6B.1|2009-11-09
    支持多种日期格式：YYYY-MM-DD, YYYY-MM, YYYY
    Returns date string or None.
    """
    if not header: 
        return None
    
    # 首先尝试找到管道符分隔的日期格式 |YYYY-MM-DD|
    pipe_date_pattern = r'\|(\d{4}-\d{2}-\d{2})\|'
    match = re.search(pipe_date_pattern, header)
    if match:
        try:
            date_str = match.group(1)
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            pass
    
    # 尝试找到管道符分隔的YYYY-MM格式 |YYYY-MM|
    pipe_date_ym_pattern = r'\|(\d{4}-\d{2})\|'
    match_ym = re.search(pipe_date_ym_pattern, header)
    if match_ym:
        try:
            date_str_ym = match_ym.group(1)
            datetime.datetime.strptime(date_str_ym + '-01', '%Y-%m-%d')
            return date_str_ym + '-01'
        except ValueError:
            pass
    
    # 尝试找到管道符分隔的YYYY格式 |YYYY|
    pipe_date_y_pattern = r'\|(\d{4})\|'
    match_y = re.search(pipe_date_y_pattern, header)
    if match_y:
        try:
            date_str_y = match_y.group(1)
            # 确保是合理的年份范围
            year = int(date_str_y)
            if 1900 <= year <= 2100:
                datetime.datetime.strptime(date_str_y + '-01-01', '%Y-%m-%d')
                return date_str_y + '-01-01'
        except ValueError:
            pass
    
    # 如果没有管道符，尝试其他常见的日期模式
    # 避免匹配序列ID中的数字，要求日期前后有适当的边界
    standalone_date_pattern = r'(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)'
    match = re.search(standalone_date_pattern, header)
    if match:
        try:
            date_str = match.group(1)
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            pass
    
    # 尝试YYYY-MM格式（要求年份在合理范围内）
    standalone_ym_pattern = r'(?<!\d)(\d{4}-\d{2})(?!\d)'
    match_ym = re.search(standalone_ym_pattern, header)
    if match_ym:
        try:
            date_str_ym = match_ym.group(1)
            year = int(date_str_ym.split('-')[0])
            if 1900 <= year <= 2100:
                datetime.datetime.strptime(date_str_ym + '-01', '%Y-%m-%d')
                return date_str_ym + '-01'
        except ValueError:
            pass
    
    # 最后尝试YYYY格式（要求年份在合理范围内，且不是序列ID的一部分）
    standalone_y_pattern = r'(?<!\d)(\d{4})(?!\d)'
    match_y = re.search(standalone_y_pattern, header)
    if match_y:
        try:
            date_str_y = match_y.group(1)
            year = int(date_str_y)
            if 1900 <= year <= 2100:
                datetime.datetime.strptime(date_str_y + '-01-01', '%Y-%m-%d')
                return date_str_y + '-01-01'
        except ValueError:
            pass
             
    return None

def calculate_n_content(sequence):
    """
    Calculate the proportion of unknown bases (N) in the sequence.
    Returns tuple (n_count, n_ratio).
    """
    if not sequence: # Handle empty sequence
        return 0, 0.0
    n_count = sequence.upper().count('N')
    try:
        n_ratio = n_count / len(sequence)
    except ZeroDivisionError:
        n_ratio = 0.0
    return n_count, n_ratio

def _process_one_filter(args):
    """Top-level worker for filter_strains (multiprocessing picklable)."""
    rec, max_n_ratio, embedded = args
    sid, hdr, s = rec
    n_count, n_ratio = calculate_n_content(s)
    if n_ratio > max_n_ratio:
        return None
    meta = None
    date_val = ''
    if embedded:
        meta = parse_header_embedded_metadata(hdr)
        date_val = meta.get('Date','')
    else:
        date_val = extract_date_from_header(hdr) or ''
    return (sid, s, n_count, n_ratio, date_val, meta)

def filter_strains(fasta_file_path: str,
                   max_n_ratio: float,
                   output_fasta_template: str,
                   output_report: str,
                   min_size: int = 0,
                   embedded: bool = False,
                   workers: int = 1) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[pd.DataFrame]]:
    """Filter sequences by N content and (optionally) harvest embedded metadata.

    Returns (filtered_fasta_path, earliest_sequence_id, date_range, metadata_df_or_None).
    metadata_df columns (embedded mode): ID, Date, Location, Lineage, Clade (Clade 固定为 'Unknown')
    """
    logging.info(f"Filtering sequences (N <= {max_n_ratio}, min_len={min_size}) ... [embedded={embedded}, workers={workers}]")

    # Collect records; 若 workers>1 则直接并行处理（不再依据阈值判定）。
    raw_records = []  # (seq_id, header, seq)
    for seq_id, header, seq in simple_fasta_parser(fasta_file_path):
        if not seq_id or not seq: continue
        if len(seq) < min_size: continue
        raw_records.append((seq_id, header, seq))

    # 使用顶层函数以便 multiprocessing 可 picklable
    def _local_call(rec):  # 单进程时的轻量包装
        return _process_one_filter((rec, max_n_ratio, embedded))

    use_parallel = workers and workers > 1
    results = []
    if use_parallel:
        from multiprocessing import Pool
        iterable = ((rec, max_n_ratio, embedded) for rec in raw_records)
        with Pool(processes=workers) as pool:
            for r in pool.imap_unordered(_process_one_filter, iterable, chunksize=200):
                if r is not None:
                    results.append(r)
    else:
        for rec in raw_records:
            r = _local_call(rec)
            if r is not None:
                results.append(r)

    if not results:
        logging.warning("No sequences passed filters")
        return None, None, None, None

    filtered = []
    quality_rows = []
    date_strings: List[str] = []
    earliest_date = None
    earliest_id = None
    embedded_meta: List[Dict[str,str]] = []
    for sid, s, n_count, n_ratio, date_val, meta in results:
        if date_val:
            date_strings.append(date_val)
            if earliest_date is None or date_val < earliest_date:
                earliest_date = date_val
                earliest_id = sid
        filtered.append((sid, s))
        quality_rows.append((sid, len(s), n_count, n_ratio))
        if embedded and meta:
            embedded_meta.append(meta)

    date_range = None
    if date_strings:
        try:
            start = min(date_strings).replace('-', '_')
            end = max(date_strings).replace('-', '_')
            date_range = f"{start}-{end}"
        except Exception as e:
            logging.warning(f"Could not compute date range: {e}")

    seq_count = len(filtered)
    out_fasta = output_fasta_template.replace('.fasta', f'_{seq_count}.fasta')
    if date_range:
        out_fasta = out_fasta.replace('.fasta', f'_{date_range}.fasta')

    try:
        with open(out_fasta, 'w') as fw:
            for sid, s in filtered:
                fw.write(f">{sid}\n{s}\n")
    except IOError as e:
        logging.error(f"Error writing filtered FASTA: {e}")
        return None, earliest_id, date_range, (pd.DataFrame(embedded_meta) if embedded_meta else None)

    try:
        pd.DataFrame(quality_rows, columns=["Accession", "Sequence_Length", "N_Count", "N_Ratio"]).to_csv(output_report, index=False)
    except Exception as e:
        logging.warning(f"Could not write quality report: {e}")

    logging.info(f"Filtered FASTA: {out_fasta}")
    logging.info(f"Quality report: {output_report}")

    meta_df = pd.DataFrame(embedded_meta) if embedded and embedded_meta else None
    return out_fasta, (earliest_id or (filtered[0][0] if filtered else None)), date_range, meta_df

def parse_sars_cov_2_mutations_with_metadata(mutations_csv_path, metadata_path, output_csv_path, metadata_format=None):
    """Vectorized join of mutations with external metadata (fewer loops, shorter code)."""
    logging.info("4. Processing mutation data and metadata information (vectorized)...")
    if metadata_format is None:
        metadata_format = os.path.splitext(metadata_path)[1].lstrip('.').lower() or 'csv'
    try:
        if metadata_format == 'tsv':
            try:
                meta_df = pd.read_csv(metadata_path, sep='\t', low_memory=False)
            except pd.errors.ParserError:
                meta_df = pd.read_csv(metadata_path, sep='\t', low_memory=False, quoting=3)
        elif metadata_format == 'csv':
            try:
                meta_df = pd.read_csv(metadata_path, low_memory=False)
            except pd.errors.ParserError:
                meta_df = pd.read_csv(metadata_path, low_memory=False, quoting=3)
        elif metadata_format in ('xls','xlsx'):
            meta_df = pd.read_excel(metadata_path)
        else:
            logging.error(f"Unsupported metadata format {metadata_format}")
            return pd.DataFrame(), None
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        return pd.DataFrame(), None
    if meta_df.empty:
        logging.warning("Metadata empty")
        return pd.DataFrame(), None
    meta_df.columns = meta_df.columns.str.strip()
    id_col = find_first_existing_column(meta_df, ['Accession ID','Isolate_Id','gisaid_epi_isl','Accession','ID'])
    if not id_col:
        logging.error("No ID column found")
        return pd.DataFrame(), None
    date_col = find_first_existing_column(meta_df, ['date','Collection_Date','Collection date'])
    lineage_col = find_first_existing_column(meta_df, ['pangolin_lineage','Lineage','Pango lineage'])
    clade_col = find_first_existing_column(meta_df, ['Clade','GISAID_clade','Nextstrain_clade'])
    if not date_col: logging.warning("Missing date column")
    if not lineage_col: logging.warning("Missing lineage column")
    if not clade_col: logging.warning("Missing clade column")
    # Location heuristic: prefer 'location', else compose
    lower_map = {c.lower(): c for c in meta_df.columns}
    loc_primary = lower_map.get('location')
    if loc_primary:
        loc_series = meta_df[loc_primary].astype(str)
    else:
        comp_cols = [lower_map.get(c) for c in ('location','division','country','region') if lower_map.get(c)]
        if comp_cols:
            loc_series = meta_df[comp_cols].astype(str).replace({'nan':'','None':''}).agg(lambda r: ' / '.join([x for x in r if x and x.lower() not in ('nan','none')]), axis=1)
        else:
            loc_series = pd.Series(['Unknown']*len(meta_df))
    meta_compact = pd.DataFrame({
        'Accession': meta_df[id_col].astype(str).str.strip(),
        'Date': meta_df[date_col].astype(str).str.strip() if date_col else '',
        'Location': loc_series.fillna('Unknown').replace({'':'Unknown'}),
        'Lineage': meta_df[lineage_col].astype(str).str.strip() if lineage_col else 'Unknown',
        'Clade': meta_df[clade_col].astype(str).str.strip() if clade_col else 'Unknown'
    })
    try:
        muts_df = pd.read_csv(mutations_csv_path)
    except Exception as e:
        logging.error(f"Error reading mutations: {e}")
        return pd.DataFrame(), None
    if muts_df.empty:
        logging.warning("Mutations empty")
        return pd.DataFrame(), None
    if 'Query_ID' not in muts_df.columns or 'Mutations' not in muts_df.columns:
        logging.error("Mutations CSV missing required columns Query_ID/Mutations")
        return pd.DataFrame(), None
    ids_df = extract_mutation_ids(muts_df)
    # Deduplicate first occurrence
    ids_df = ids_df.drop_duplicates('Accession')
    merged = ids_df.merge(meta_compact, on='Accession', how='inner')
    if merged.empty:
        logging.warning("No overlap between mutations and metadata")
        return pd.DataFrame(), None
    merged.rename(columns={'Accession':'ID','Mutations':'Mutations_str'}, inplace=True)
    merged['Date'] = normalize_dates(merged['Date'])
    date_range = compute_date_range(merged['Date'])
    try:
        merged[['ID','Date','Location','Lineage','Clade','Mutations_str']].to_csv(output_csv_path, sep='\t', index=False)
    except Exception as e:
        logging.error(f"Error writing output CSV: {e}")
    return merged[['ID','Date','Location','Lineage','Clade','Mutations_str']], date_range

def find_fasta_file(input_dir):
    """
    Find a FASTA file in the given input directory.
    Prioritizes .fasta, then .fa, then .fna. Returns the first one found.
    """
    fasta_extensions = ['.fasta', '.fa', '.fna']
    for ext in fasta_extensions:
        # Use iglob for potentially large directories (memory efficient)
        files = glob.glob(os.path.join(input_dir, f"*{ext}"))
        if files:
            # Maybe add logic here if multiple files are found? For now, return first.
            return files[0]
    return None

def find_metadata_file(input_dir):
    """
    Find a metadata file in the given input directory.
    Supported formats: .tsv, .xlsx, .xls, .csv (priority order)
    Returns tuple of (file_path, file_format) or (None, None).
    """
    extensions = ['.tsv', '.xlsx', '.xls', '.csv']
    for ext in extensions:
        metadata_files = glob.glob(os.path.join(input_dir, f"*{ext}"))
        if metadata_files:
            # Maybe add logic here if multiple files are found? For now, return first.
            return metadata_files[0], ext.lstrip('.')
    return None, None

def integrate_mutations_with_embedded_meta(mutations_csv_path: str,
                                           metadata_df: pd.DataFrame,
                                           output_csv_path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """向量化整合变异与内嵌元数据。

    需要 metadata_df 列: ID, Date, Location, Lineage, Clade
    返回 (DataFrame, date_range)；失败返回 (空DF, None)。"""
    try:
        muts_df = pd.read_csv(mutations_csv_path)
    except Exception as e:
        logging.error(f"Cannot read mutations CSV: {e}")
        return pd.DataFrame(), None
    if metadata_df is None or metadata_df.empty:
        logging.warning("Embedded metadata DataFrame empty; skipping merge.")
        return pd.DataFrame(), None
    if 'Query_ID' not in muts_df.columns:
        logging.error("Mutations CSV missing 'Query_ID' column")
        return pd.DataFrame(), None
    muts_sub = muts_df[['Query_ID','Mutations']].copy()
    muts_sub.rename(columns={'Query_ID':'ID','Mutations':'Mutations_str'}, inplace=True)
    meta_compact = metadata_df[['ID','Date','Location','Lineage','Clade']].copy()
    merged = muts_sub.merge(meta_compact, on='ID', how='inner')
    if merged.empty:
        logging.warning("No overlap between mutations and embedded metadata IDs")
        return pd.DataFrame(), None
    merged['Date'] = normalize_dates(merged['Date'])
    out_df = merged[['ID','Date','Location','Lineage','Clade','Mutations_str']]
    date_range = compute_date_range(out_df['Date'])
    try:
        out_df.to_csv(output_csv_path, sep='\t', index=False)
    except Exception as e:
        logging.warning(f"Failed writing embedded merged csv: {e}")
    return out_df, date_range

def run_pipeline(input_dir: str,
                 output_dir: str,
                 ratio: float = 0.001,
                 ref: Optional[str] = None,
                 n: int = 4,
                 min_size: int = 0,
                 force_mode: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """执行完整流程（自动检测内嵌 / 外部元数据模式）。

    返回 (final_df, final_csv_path)；失败返回 (None, None)。"""
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isdir(input_dir):
        fasta_file = find_fasta_file(input_dir)
        if not fasta_file:
            logging.error(f"No FASTA file in directory {input_dir}")
            return None, None
        metadata_file, metadata_format = find_metadata_file(input_dir)
    else:
        fasta_file = input_dir
        if not os.path.isfile(fasta_file):
            logging.error(f"FASTA file not found: {fasta_file}")
            return None, None
        metadata_file, metadata_format = (None, None)

    mode = force_mode if force_mode in ('embedded','external') else ('external' if metadata_file else 'embedded')
    logging.info(f"Detected mode: {mode} (metadata_file={'present' if metadata_file else 'absent'})")

    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    filtered_template = os.path.join(output_dir, f"{base_name}_filtered_N_lt_{ratio}.fasta")
    quality_csv = os.path.join(output_dir, f"{base_name}_quality_report_{ratio}.csv")
    processed_base = os.path.join(output_dir, f"{base_name}_processed_data")

    logging.info("===== STEP 1.1: Filter & Collect Metadata =====")
    filtered_fasta, earliest_id, date_range, embedded_meta_df = filter_strains(
        fasta_file, ratio, filtered_template, quality_csv, min_size, embedded=(mode=='embedded'), workers=n
    )
    if not filtered_fasta:
        logging.error("Aborting: filtering failed")
        return None, None

    if ref is None:
        ref = earliest_id
        if not ref:
            logging.warning("No reference sequence chosen; variant step may fail")

    logging.info("===== STEP 1.2: Alignment (halign4) =====")
    halign_base = os.path.join(output_dir, f"{base_name}_halign4")
    halign_out = f"{halign_base}.fasta" if not date_range else f"{halign_base}_{date_range}.fasta"
    try:
        subprocess.run(f"halign4 {filtered_fasta} {halign_out} -t {n}", shell=True, check=True)
    except FileNotFoundError:
        logging.error("halign4 not in PATH")
        return None, None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running halign4: {e}")
        return None, None
    logging.info(f"Alignment output: {halign_out}")

    logging.info("===== STEP 1.3: Variant Marking =====")
    variant_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'variant_mark_ljj.py')
    if not os.path.exists(variant_script):
        variant_script = 'variant_mark_ljj.py'
    # variant_mark_ljj.py 当前参数不支持 --base_name，改为固定输出文件名处理
    variant_cmd = f"python {variant_script} -fas {halign_out} -ref {ref} -o {output_dir} -t {n}"
    try:
        subprocess.run(variant_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running variant_mark_ljj.py: {e}")
        return None, None

    # 直接使用脚本固定输出文件名（保持简洁）
    mutations_csv = os.path.join(output_dir, "mutations_result.csv")

    logging.info("===== STEP 1.4: Merge Mutations + Metadata =====")
    output_csv_base = f"{processed_base}.csv"
    if mode=='external':
        if not metadata_file:
            logging.error("External mode but no metadata file found")
            return None, None
        final_df, meta_range = parse_sars_cov_2_mutations_with_metadata(
            mutations_csv_path=mutations_csv,
            metadata_path=metadata_file,
            output_csv_path=output_csv_base,
            metadata_format=metadata_format
        )
    else:
        if embedded_meta_df is None or embedded_meta_df.empty:
            final_df, meta_range = pd.DataFrame(), date_range
        else:
            final_df, meta_range = integrate_mutations_with_embedded_meta(
                mutations_csv_path=mutations_csv,
                metadata_df=embedded_meta_df,
                output_csv_path=output_csv_base
            )
    if final_df is None:
        return None, None
    final_range = meta_range or date_range
    if final_range and os.path.exists(output_csv_base):
        ranged = f"{processed_base}_{final_range}.csv"
        try:
            os.replace(output_csv_base, ranged)
            logging.info(f"Renamed final output with date range: {ranged}")
        except Exception as e:
            logging.warning(f"Rename failed: {e}")

    logging.info("Pipeline completed successfully.")
    # 寻找最终 processed_data 输出文件
    final_csv_path = None
    try:
        candidates = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv') and '_processed_data' in f]
        if candidates:
            final_csv_path = max(candidates, key=os.path.getmtime)
    except Exception as e:
        logging.warning(f"Failed locating final processed csv: {e}")
    return final_df, final_csv_path



def main():
    parser = argparse.ArgumentParser(description='Unified sequence processing pipeline (external or embedded metadata).')
    parser.add_argument('--input', required=True, help='Input directory containing FASTA (+ optional metadata) OR a single FASTA file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--ratio', type=float, default=0.001, help='Maximum N base ratio (default: 0.001)')
    parser.add_argument('--min_size', type=int, default=0, help='Minimum sequence length to retain (default: 0)')
    parser.add_argument('--ref', default=None, help='Reference sequence ID (default: earliest dated sequence)')
    parser.add_argument('-n', type=int, default=4, help='Processes / threads for alignment & variant marking')
    parser.add_argument('--mode', choices=['embedded','external'], help='Force mode (skip auto-detect)')

    args = parser.parse_args()

    df, _ = run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        ratio=args.ratio,
        ref=args.ref,
        n=args.n,
        min_size=args.min_size,
        force_mode=args.mode
    )
    if df is None:
        logging.info("Finished with errors or empty result.")
    else:
        logging.info(f"Finished. Records: {len(df)}")


if __name__ == "__main__":
    main()