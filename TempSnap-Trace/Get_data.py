import os
import argparse
import subprocess
import re
import pandas as pd
import glob
import sys
import datetime
from typing import Tuple, Optional, Iterator, Dict, Any, List

def simple_fasta_parser(fasta_filename: str) -> Iterator[Tuple[str, str, str]]:
    """
    A simple generator function to parse FASTA files without Biopython.
    Yields tuples of (id, description, sequence).
    Handles potential file errors.
    """
    sequence = ''
    header = None
    seq_id = None
    description = None

    try:
        with open(fasta_filename, 'r') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        yield (seq_id, description, sequence)
                    
                    header = line[1:]
                    parts = header.split(None, 1)
                    seq_id = parts[0]
                    description = header
                    sequence = ''
                elif header is not None:
                    sequence += line
            
            if header is not None:
                yield (seq_id, description, sequence)
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_filename}", file=sys.stderr)
    except Exception as e:
        print(f"Error parsing FASTA file {fasta_filename}: {e}", file=sys.stderr)

def extract_date_from_header(header: Optional[str]) -> Optional[str]:
    """
    Extracts date (YYYY-MM-DD format) from sequence header.
    Prioritizes YYYY-MM-DD, falls back to YYYY-MM (as YYYY-MM-01).
    Returns date string or None.
    """
    if not header: return None
    
    # Try YYYY-MM-DD format
    date_pattern_ymd = r'(\d{4}-\d{2}-\d{2})'
    match_ymd = re.search(date_pattern_ymd, header)
    if match_ymd:
        try:
            date_str = match_ymd.group(1)
            datetime.datetime.strptime(date_str, '%Y-%m-%d') 
            return date_str
        except ValueError:
            pass # Invalid date like 2020-00-00
            
    # Fallback: Try YYYY-MM format (treat as first day of month)
    date_pattern_ym = r'(\d{4}-\d{2})'
    match_ym = re.search(date_pattern_ym, header)
    if match_ym:
        try:
            date_str_ym = match_ym.group(1)
            # Ensure it's not part of a YYYY-MM-DD pattern
            if not re.search(r'\d{4}-\d{2}-', header): # Check if followed by '-'
                 datetime.datetime.strptime(date_str_ym + '-01', '%Y-%m-%d')
                 return date_str_ym + '-01'
        except ValueError:
            pass
             
    return None

def calculate_n_content(sequence: str) -> Tuple[int, float]:
    """
    Calculate the proportion of unknown bases (N) in the sequence.
    Returns tuple (n_count, n_ratio).
    """
    if not sequence:
        return 0, 0.0
    n_count = sequence.upper().count('N')
    seq_len = len(sequence)
    n_ratio = n_count / seq_len if seq_len > 0 else 0.0
    return n_count, n_ratio

def filter_strains(fasta_file_path: str, max_n_ratio: float, 
                   output_fasta_template: str, output_report: str, 
                   min_size: int = 0) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Filter strains based on N content and minimum length using pure Python parser.
    Returns (output_fasta_path, earliest_seq_id, date_range_str).
    """
    print(f"Filtering sequences with N ratio <= {max_n_ratio} and length >= {min_size}...")
    filtered_sequences_data = [] # Store tuples (id, sequence)
    quality_report = []         # Store tuples for CSV report
    all_valid_dates = []        # Store valid date strings (YYYY-MM-DD)
    
    earliest_date_str: Optional[str] = None
    earliest_seq_id: Optional[str] = None

    for seq_id, description, sequence in simple_fasta_parser(fasta_file_path):
        if not seq_id or not sequence:
            continue
            
        if len(sequence) < min_size:
            continue

        n_count, n_ratio = calculate_n_content(sequence)
        
        if n_ratio > max_n_ratio:
            continue

        seq_date_str = extract_date_from_header(description) 
        if seq_date_str:
            all_valid_dates.append(seq_date_str)
            if earliest_date_str is None or seq_date_str < earliest_date_str:
                earliest_date_str = seq_date_str
                earliest_seq_id = seq_id 

        filtered_sequences_data.append((seq_id, sequence)) 
        quality_report.append((seq_id, len(sequence), n_count, n_ratio)) 

    date_range_str: Optional[str] = None
    if all_valid_dates:
        try:
            start_date_str = min(all_valid_dates)
            end_date_str = max(all_valid_dates)
            date_range_str = f"{start_date_str.replace('-', '_')}-{end_date_str.replace('-', '_')}"
        except Exception as e:
            print(f"Warning: Could not determine date range from collected dates: {e}", file=sys.stderr)
    
    sequence_count = len(filtered_sequences_data)
    output_fasta = output_fasta_template.replace('.fasta', f'_{sequence_count}.fasta')
    if date_range_str:
        output_fasta = output_fasta.replace('.fasta', f'_{date_range_str}.fasta')

    try:
        with open(output_fasta, 'w') as f:
            for seq_id, sequence in filtered_sequences_data:
                if seq_id and sequence:
                     f.write(f">{seq_id}\n{sequence}\n")
    except IOError as e:
        print(f"Error writing filtered FASTA file {output_fasta}: {e}", file=sys.stderr)
        return None, earliest_seq_id, date_range_str # Indicate failure

    try:
        df = pd.DataFrame(quality_report, columns=["Accession", "Sequence_Length", "N_Count", "N_Ratio"])
        df.to_csv(output_report, index=False)
    except Exception as e:
         print(f"Error writing quality report CSV {output_report}: {e}", file=sys.stderr)

    print(f"Filtered {sequence_count} sequences saved to: {output_fasta}")
    print(f"Quality report saved to: {output_report}")
    
    return output_fasta, earliest_seq_id, date_range_str

def _find_column_ignore_case(df_columns: pd.Index, possible_names: List[str]) -> Optional[str]:
    """Helper to find the first matching column name, case-insensitive."""
    df_columns_lower = {col.lower(): col for col in df_columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None

def _parse_date_robust(date_str: Any) -> pd.Timestamp:
    """Robustly parse date strings into Timestamps."""
    if pd.isna(date_str) or str(date_str).strip() == '':
        return pd.NaT
    date_str = str(date_str).strip()
    # Try formats in order of preference
    formats_to_try = ['%Y-%m-%d', '%Y-%m', '%Y']
    for fmt in formats_to_try:
        try:
            # Handle partial dates by appending defaults
            if fmt == '%Y-%m':
                return pd.to_datetime(date_str + '-01', format='%Y-%m-%d', errors='raise')
            elif fmt == '%Y':
                return pd.to_datetime(date_str + '-01-01', format='%Y-%m-%d', errors='raise')
            else: # YYYY-MM-DD
                return pd.to_datetime(date_str, format=fmt, errors='raise')
        except ValueError:
            continue # Try next format
    # print(f"Warning: Could not parse date '{date_str}' with known formats.")
    return pd.NaT # Return Not-a-Time if all formats fail

def parse_sars_cov_2_mutations_with_metadata(mutations_csv_path: str, metadata_path: str, 
                                             output_csv_path: str, 
                                             metadata_format: Optional[str] = None
                                             ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read mutation data and metadata, match by ID, and save merged data.
    Returns (DataFrame, date_range_str) or (None, None) on error.
    """
    print(f"Processing mutation data and metadata...")
    
    if metadata_format is None:
        metadata_format = os.path.splitext(metadata_path)[1].lstrip('.').lower() or 'csv'

    # Read metadata
    try:
        read_opts = {'low_memory': False}
        if metadata_format == 'tsv':
            try:
                 metadata_df = pd.read_csv(metadata_path, sep='\t', **read_opts)
            except pd.errors.ParserError:
                 print("Warning: TSV parsing error, trying with quoting=3 (QUOTE_NONE)", file=sys.stderr)
                 metadata_df = pd.read_csv(metadata_path, sep='\t', quoting=3, **read_opts)
        elif metadata_format == 'csv':
             try:
                 metadata_df = pd.read_csv(metadata_path, **read_opts)
             except pd.errors.ParserError:
                 print("Warning: CSV parsing error, trying with quoting=3 (QUOTE_NONE)", file=sys.stderr)
                 metadata_df = pd.read_csv(metadata_path, quoting=3, **read_opts)
        elif metadata_format in ['xls', 'xlsx']:
            metadata_df = pd.read_excel(metadata_path)
        else:
            print(f"Error: Unsupported metadata format: {metadata_format}", file=sys.stderr)
            return None, None
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_path}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}", file=sys.stderr)
        return None, None
        
    print(f"Loaded metadata with {len(metadata_df)} entries")
    metadata_df.columns = metadata_df.columns.str.strip() # Clean column names

    # Find key columns
    id_column = _find_column_ignore_case(metadata_df.columns, 
                                         ['Accession ID', 'Isolate_Id', 'gisaid_epi_isl', 'Accession', 'ID'])
    date_column = _find_column_ignore_case(metadata_df.columns, 
                                           ['date', 'Collection_Date', 'Collection date'])
    lineage_column = _find_column_ignore_case(metadata_df.columns, 
                                              ['pangolin_lineage', 'Lineage', 'Pango lineage'])
    clade_column = _find_column_ignore_case(metadata_df.columns, 
                                            ['Clade', 'GISAID_clade', 'Nextstrain_clade'])
    location_column = _find_column_ignore_case(metadata_df.columns, ['Location', 'location'])
    country_column = _find_column_ignore_case(metadata_df.columns, ['Country', 'country']) # Fallback for location

    if not id_column:
        print("Error: No suitable ID column found in metadata.", file=sys.stderr)
        return None, None
    if not date_column: print("Warning: No date column found in metadata.", file=sys.stderr)
    if not lineage_column: print("Warning: No lineage column found.", file=sys.stderr)
    if not clade_column: print("Warning: No clade column found.", file=sys.stderr)
    if not location_column and not country_column: print("Warning: No location or country column found.", file=sys.stderr)
    
    print(f"Using ID column: '{id_column}'")
    if date_column: print(f"Using Date column: '{date_column}'")
    if lineage_column: print(f"Using Lineage column: '{lineage_column}'")
    if clade_column: print(f"Using Clade column: '{clade_column}'")
    if location_column: print(f"Using Location column: '{location_column}'")
    elif country_column: print(f"Using Country column as fallback location: '{country_column}'")


    # Create metadata mapping dictionary
    metadata_mapping: Dict[str, Dict[str, Any]] = {}
    metadata_df[id_column] = metadata_df[id_column].astype(str).str.strip()
    
    # Use vectorization for faster lookup creation if possible, fallback to iterrows
    try:
        metadata_df_indexed = metadata_df.drop_duplicates(subset=[id_column]).set_index(id_column)
        for acc_id, row in metadata_df_indexed.iterrows():
            if not acc_id: continue
            
            location = "Unknown"
            if location_column and not pd.isna(row[location_column]):
                location = str(row[location_column]).strip()
            elif country_column and not pd.isna(row[country_column]): # Fallback
                location = str(row[country_column]).strip()

            metadata_mapping[acc_id] = {
                'Collection_Date': str(row[date_column]).strip() if date_column and not pd.isna(row[date_column]) else '',
                'Location': location,
                'Lineage': str(row[lineage_column]).strip() if lineage_column and not pd.isna(row[lineage_column]) else 'Unknown',
                'Clade': str(row[clade_column]).strip() if clade_column and not pd.isna(row[clade_column]) else 'Unknown'
            }
    except Exception as e: # Fallback to iterrows if indexing fails (e.g., non-unique IDs not dropped)
         print(f"Warning: Optimized metadata processing failed ({e}), falling back to iterrows.", file=sys.stderr)
         processed_ids = set()
         for _, row in metadata_df.iterrows():
             acc_id = row[id_column]
             if not acc_id or acc_id in processed_ids: continue
             processed_ids.add(acc_id)
             
             location = "Unknown"
             if location_column and not pd.isna(row[location_column]):
                 location = str(row[location_column]).strip()
             elif country_column and not pd.isna(row[country_column]): # Fallback
                 location = str(row[country_column]).strip()

             metadata_mapping[acc_id] = {
                 'Collection_Date': str(row[date_column]).strip() if date_column and not pd.isna(row[date_column]) else '',
                 'Location': location,
                 'Lineage': str(row[lineage_column]).strip() if lineage_column and not pd.isna(row[lineage_column]) else 'Unknown',
                 'Clade': str(row[clade_column]).strip() if clade_column and not pd.isna(row[clade_column]) else 'Unknown'
             }

    # Read and process mutation data
    try:
        mutations_df = pd.read_csv(mutations_csv_path)
    except FileNotFoundError:
        print(f"Error: Mutations CSV file not found at {mutations_csv_path}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Error reading mutations CSV file {mutations_csv_path}: {e}", file=sys.stderr)
        return None, None

    data = []
    processed_mutation_ids = set()
    for _, row in mutations_df.iterrows():
        if 'Query_ID' not in row or 'Mutations' not in row:
             continue
             
        query_id = str(row['Query_ID']).strip()
        mutations = str(row['Mutations']).strip() if not pd.isna(row['Mutations']) else ''
        
        parts = query_id.split('|')
        acc_id = parts[1].strip() if len(parts) >= 2 else None
        
        if not acc_id or acc_id in processed_mutation_ids:
             continue
        processed_mutation_ids.add(acc_id)

        if acc_id in metadata_mapping:
            metadata = metadata_mapping[acc_id]
            data.append({
                'ID': acc_id,
                'Date': metadata['Collection_Date'],
                'Location': metadata['Location'],
                'Lineage': metadata['Lineage'],
                'Clade': metadata['Clade'],
                'Mutations_str': mutations
            })

    if not data:
         print("Warning: No matching records found between mutations and metadata.", file=sys.stderr)
         return pd.DataFrame(), None # Return empty DataFrame
         
    result_df = pd.DataFrame(data)
    result_df['Date'] = result_df['Date'].apply(_parse_date_robust)
    
    # Get date range from the final DataFrame
    date_range_str: Optional[str] = None
    try:
        valid_dates = result_df['Date'].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().strftime('%Y_%m_%d')
            max_date = valid_dates.max().strftime('%Y_%m_%d')
            date_range_str = f"{min_date}-{max_date}"
    except Exception as e:
        print(f"Warning: Could not determine date range from final data: {e}", file=sys.stderr)
    
    # Save DataFrame using tab delimiter
    try:
        result_df.to_csv(output_csv_path, sep='\t', header=True, index=False)
    except Exception as e:
        print(f"Error writing final processed data CSV {output_csv_path}: {e}", file=sys.stderr)
        # Don't return None here, the DataFrame might still be useful

    return result_df, date_range_str

def find_file_by_extensions(input_dir: str, extensions: List[str]) -> Optional[str]:
    """Find the first file matching any of the extensions in the directory."""
    for ext in extensions:
        files = glob.glob(os.path.join(input_dir, f"*{ext}"))
        if files:
            return files[0] # Return the first match
    return None

def run_pipeline(input_dir: str, output_dir: str, ratio: float = 0.001, 
                 ref: Optional[str] = None, n: int = 4, min_size: int = 0
                 ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute the complete processing pipeline.
    Returns (final_dataframe, final_output_path) or (None, None) on critical error.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
        return None, None

    # Find input files
    fasta_file = find_file_by_extensions(input_dir, ['.fasta', '.fa', '.fna'])
    if not fasta_file:
        print(f"Error: No FASTA file found in {input_dir}", file=sys.stderr)
        return None, None
    print(f"Found FASTA file: {fasta_file}")
    
    metadata_file = find_file_by_extensions(input_dir, ['.tsv', '.xlsx', '.xls', '.csv'])
    if not metadata_file:
        print(f"Error: No metadata file found in {input_dir}", file=sys.stderr)
        return None, None
    metadata_format = os.path.splitext(metadata_file)[1].lstrip('.')
    print(f"Found metadata file: {metadata_file} (format: {metadata_format})")
    
    # Define base names and paths
    base_filename = os.path.splitext(os.path.basename(fasta_file))[0]
    filtered_fasta_template = os.path.join(output_dir, f"{base_filename}_filtered_N_lt_{ratio}.fasta")
    quality_report_path = os.path.join(output_dir, f"{base_filename}_quality_report_{ratio}.csv")
    processed_data_base = os.path.join(output_dir, f"{base_filename}_processed_data") # Base for final CSV
    halign_output_base = os.path.join(output_dir, f"{base_filename}_halign4") # Base for alignment
    mutations_csv = os.path.join(output_dir, f"{base_filename}_mutations_result.csv") # Expected variant output
    stats_csv = os.path.join(output_dir, f"{base_filename}_mutations_stats.csv") # Expected variant stats output

    # --- STEP 1: Filtering Strains ---
    print("\n===== STEP 1: Filtering Strains =====")
    filter_result = filter_strains(fasta_file, ratio, filtered_fasta_template, quality_report_path, min_size)
    if filter_result is None:
         print("Error during strain filtering. Aborting pipeline.", file=sys.stderr)
         return None, None
    filtered_fasta, earliest_seq_id, filter_date_range = filter_result
    if not filtered_fasta or not os.path.exists(filtered_fasta):
         print("Error: Filtered FASTA file not created or path invalid. Aborting pipeline.", file=sys.stderr)
         return None, None

    # Determine reference sequence
    if ref is None:
        if earliest_seq_id:
            ref = earliest_seq_id
            print(f"Using earliest sequence as reference: {ref}")
        else:
            print("Error: No reference specified and could not determine earliest sequence.", file=sys.stderr)
            return None, None # Reference is required for variant marking
    
    # Construct alignment output path with date range from filtering
    halign_output = f"{halign_output_base}.fasta"
    if filter_date_range:
        halign_output = f"{halign_output_base}_{filter_date_range}.fasta"
    
    # --- STEP 2: Multiple Sequence Alignment (halign4) ---
    print(f"\n===== STEP 2: Multiple Sequence Alignment (halign4) =====")
    halign_cmd = f"halign4 {filtered_fasta} {halign_output} -t {n}"
    try:
        print(f"Running command: {halign_cmd}")
        result = subprocess.run(halign_cmd, shell=True, check=True, capture_output=True, text=True)
        print("halign4 completed successfully.")
        print(f"Alignment saved to: {halign_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing halign4: {e}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return None, None
    except FileNotFoundError:
         print("Error: 'halign4' command not found. Ensure it is installed and in PATH.", file=sys.stderr)
         return None, None

    # --- STEP 3: Marking Variant Positions ---
    print(f"\n===== STEP 3: Marking Variant Positions =====")
    variant_script_path = "variant_mark_ljj.py" # Assumed relative path or in PATH
    variant_cmd = f"python {variant_script_path} -fas \"{halign_output}\" -ref \"{ref}\" -o \"{output_dir}\" -t {n} --base_name \"{base_filename}\""
    try:
        print(f"Running command: {variant_cmd}")
        result = subprocess.run(variant_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=None, text=True)
        print("Variant marking completed successfully.")
        # Check if expected output exists
        if not os.path.exists(mutations_csv):
             print(f"Warning: Expected mutations output file not found: {mutations_csv}", file=sys.stderr)
             # Decide if this is critical - maybe allow continuation? For now, continue.
    except subprocess.CalledProcessError as e:
        print(f"Error executing variant marking script: {e}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return None, None
    except FileNotFoundError:
         print(f"Error: Variant marking script '{variant_script_path}' not found.", file=sys.stderr)
         return None, None

    # --- STEP 4: Parsing Mutations and Metadata ---
    print(f"\n===== STEP 4: Parsing Mutations and Metadata =====")
    final_output_path = f"{processed_data_base}.csv" # Initial path without date range
    
    parse_result = parse_sars_cov_2_mutations_with_metadata(
        mutations_csv_path=mutations_csv,
        metadata_path=metadata_file,
        output_csv_path=final_output_path, # Save to base name first
        metadata_format=metadata_format
    )
    
    if parse_result is None or parse_result[0] is None:
         print("Error during mutation/metadata parsing. Aborting pipeline.", file=sys.stderr)
         return None, None
         
    final_df, meta_date_range = parse_result
    
    # Rename the final file with date range if available
    final_output_path_with_range = final_output_path # Default to base path
    if meta_date_range:
        target_path = f"{processed_data_base}_{meta_date_range}.csv"
        try:
            if os.path.exists(final_output_path):
                 os.rename(final_output_path, target_path)
                 final_output_path_with_range = target_path # Update path if rename succeeds
                 print(f"Final processed data saved to: {final_output_path_with_range}")
            else:
                 print(f"Warning: Base processed data file not found, cannot rename: {final_output_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not rename file with date range: {e}", file=sys.stderr)
            # Keep final_output_path_with_range as the base path
    else:
         print(f"Final processed data saved to: {final_output_path}") # No date range available

    print("\nData processing pipeline completed successfully!")
    return final_df, final_output_path_with_range


def main():
    parser = argparse.ArgumentParser(description='Sequence Analysis Pipeline')
    parser.add_argument('--input_dir', required=True, help='Input directory containing FASTA and metadata files')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--ratio', type=float, default=0.001, help='Maximum N base ratio for filtering (default: 0.001)')
    parser.add_argument('--min_size', type=int, default=0, help='Minimum sequence length for filtering (default: 0, no filtering)')
    parser.add_argument('--ref', default=None, help='Reference sequence ID for variant marking (default: uses earliest sequence found)')
    parser.add_argument('--n', type=int, default=4, help='Number of threads/processes for parallel tasks (default: 4)')
    
    args = parser.parse_args()
    
    result_df, final_path = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ratio=args.ratio,
        ref=args.ref,
        n=args.n,
        min_size=args.min_size
    )
    
    if result_df is not None:
         print(f"Pipeline finished successfully. Final output: {final_path}")
         sys.exit(0)
    else:
         print("Pipeline finished with errors.", file=sys.stderr)
         sys.exit(1)


if __name__ == "__main__":
    main()