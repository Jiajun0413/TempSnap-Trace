#Author: liujiajun
#Date: 2025-10-19
#Description: 输入多序列比对文件，指定参考序列ID，鉴定序列中变异位点数据
import argparse
import multiprocessing as mp
import os
import sys
import pandas as pd
import numpy as np
import numba
import logging
from numba.core import types
from numba.typed import List
from Bio import SeqIO
from typing import Optional, Dict, Tuple

# --- Global Constants ---
WRITE_BUFFER_SIZE = 8192

# --- Data Representation ---
INT_MAP_ARR = np.full(256, -1, dtype=np.int8)
_BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4, 'N': 5, 'X': 5}
for char, code in _BASE_MAP.items():
    INT_MAP_ARR[ord(char)] = code
    INT_MAP_ARR[ord(char.lower())] = code
GAP_CODE, NX_CODE = 4, 5

# --- Worker Globals ---
G_REFSEQ_INT: Optional[np.ndarray] = None
G_REF_ACC: Optional[str] = None
G_REF_LEN: Optional[int] = None

def _init_worker(refseq_int: np.ndarray, ref_acc: str, ref_length: int):
    global G_REFSEQ_INT, G_REF_ACC, G_REF_LEN
    G_REFSEQ_INT, G_REF_ACC, G_REF_LEN = refseq_int, ref_acc, ref_length

@numba.jit(nopython=True, cache=True)
def _core_comparison_numba(ref_int: np.ndarray, que_int: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    counts = np.zeros(4, dtype=np.int32)
    variants_str = List.empty_list(types.unicode_type)
    n, P1, P2, i = len(ref_int), 0, 0, 0
    que_clone = que_int.copy()
    
    def to_char(code: int) -> str:
        if code == 0: return 'A'
        if code == 1: return 'C'
        if code == 2: return 'G'
        if code == 3: return 'T'
        if code == 4: return '-'
        if code == 5: return 'N'
        if code == 6: return 'X'
        return '?'

    while i < n:
        rc, qc = ref_int[i], que_clone[i]
        if 0 <= rc <= 3: P1 += 1
        if qc != GAP_CODE: P2 += 1
        if qc == NX_CODE: qc = rc; que_clone[i] = rc
        if rc != qc:
            if rc == GAP_CODE:
                ref_base_code, que_base_code = (ref_int[0], que_clone[0]) if P1 == 0 else (ref_int[i-1], que_clone[i-1])
                vtype = "Insertion" if ref_base_code == que_base_code else "Indel"
                vtype_idx = 1 if ref_base_code == que_base_code else 3
                start = 1 if P1 == 0 else P1
                j = i
                while j < n and ref_int[j] == GAP_CODE: j += 1
                run_len, end_j = j - i, j
                ref_allele_char, que_allele_str = "", ""
                if P1 == 0:
                    ref_allele_char = to_char(ref_int[0])
                    que_allele_str = to_char(que_clone[0])
                    for k in range(i + 1, end_j): que_allele_str += to_char(que_clone[k])
                    if end_j < n and ref_int[end_j] != GAP_CODE:
                        ref_allele_char, que_allele_str = to_char(ref_int[end_j]), que_allele_str + to_char(que_clone[end_j])
                else:
                    ref_allele_char = to_char(ref_int[i-1])
                    que_allele_str = to_char(que_clone[i-1])
                    for k in range(i, end_j): que_allele_str += to_char(que_clone[k])
                if P1 == 0 and end_j < n and ref_int[end_j] != GAP_CODE: i = end_j; P1 += 1; P2 += 1
                else: i = end_j - 1
                P2 += (run_len - 1); counts[vtype_idx] += 1
                variants_str.append(f"{start}({vtype}:{ref_allele_char}->{que_allele_str})")
            elif qc == GAP_CODE:
                vtype_tmp_diff = False
                if P2 != 0: vtype_tmp_diff = (ref_int[i-1] != que_clone[i-1])
                vtype = "Indel" if P2 != 0 and vtype_tmp_diff else "Deletion"
                vtype_idx = 3 if P2 != 0 and vtype_tmp_diff else 2
                start, que_allele_char = (1, to_char(que_clone[0])) if P2 == 0 else (P1 - 1, to_char(que_clone[i-1]))
                ref_allele_str = to_char(ref_int[i-1]) if P2 != 0 else to_char(ref_int[0])
                j = i
                while j < n and que_clone[j] == GAP_CODE:
                    ref_allele_str += to_char(ref_int[j]); j += 1
                run_len, i = j - i, j - 1
                end = P1 + run_len - 1
                counts[vtype_idx] += 1
                variants_str.append(f"{start}({vtype}:{ref_allele_str}->{que_allele_char})"); P1 = end
            else:
                k = i + 1
                if k < n and (ref_int[k] == GAP_CODE or que_clone[k] == GAP_CODE): i += 1; continue
                counts[0] += 1
                variants_str.append(f"{P1}(SNP:{to_char(rc)}->{to_char(qc)})")
        i += 1
    return counts, variants_str

def process_sequence_worker(task: Tuple[str, str]) -> Tuple[Dict, Dict]:
    key, queseq_str = task
    assert G_REFSEQ_INT is not None and G_REF_ACC is not None and G_REF_LEN is not None
    ref_int, ref_length = G_REFSEQ_INT, G_REF_LEN
    ascii_arr = np.frombuffer(queseq_str.encode('ascii', 'ignore'), dtype=np.uint8)
    queseq_int = INT_MAP_ARR[ascii_arr]
    query_length = np.count_nonzero(queseq_int != GAP_CODE)

    counts_arr, details_list = _core_comparison_numba(ref_int, queseq_int)
    details_str = ';'.join(details_list)
    
    counts = {'SNP': int(counts_arr[0]), 'Insertion': int(counts_arr[1]), 'Deletion': int(counts_arr[2]), 'Indel': int(counts_arr[3])}
    TotalVarNo = counts_arr.sum(); denom = (ref_length + query_length)
    r_val = 0 if denom == 0 else (1 - 2 * TotalVarNo / denom) * 100
    stats = {'Ref_ID': G_REF_ACC, 'Ref_length': ref_length, 'Query_ID': key, 'Query_length': query_length, 'SNP#': counts['SNP'], 'Insertion#': counts['Insertion'], 'Deletion#': counts['Deletion'], 'Indel#': counts['Indel'], 'Similarity': f"{r_val:.2f}"}
    muts = {'Query_ID': key, 'Mutations': details_str}
    return stats, muts

def find_reference_record(ref_input: str, fas_file: str) -> Tuple[str, str]:
    ref_id_str = ref_input.strip('"\'')
    matching_records = []
    for record in SeqIO.parse(fas_file, "fasta"):
        if ref_id_str in record.description:
            matching_records.append(record)
    if not matching_records: raise ValueError(f"Reference '{ref_id_str}' not found.")
    record = matching_records[0]
    if len(matching_records) > 1: logging.warning(f"Multiple matches for '{ref_id_str}'. Using first: '{record.description}'.")
    else: logging.info(f"Reference found: '{record.description}'")
    return record.description, str(record.seq)

def main():
    parser = argparse.ArgumentParser(description="Variant Analysis Tool (v.Definitive-Finale).")
    parser.add_argument("-fas", required=True, help="Input FASTA file")
    parser.add_argument("-ref", required=True, help="Reference ID (full or partial)")
    parser.add_argument("-o", required=True, help="Output directory")
    parser.add_argument("-t", type=int, default=os.cpu_count(), help=f"Number of processes (default: {os.cpu_count()})")
    args = parser.parse_args()
    
    # Configure logging
    import TempSnap
    TempSnap.LogManager.configure_logging(args.o, process_type="variant_analysis")

    logging.info("Finding reference sequence...")
    ref_acc, ref_seq_str = find_reference_record(args.ref, args.fas)
    ref_length = len(ref_seq_str.replace('-', ''))

    tasks = [(rec.description, str(rec.seq)) for rec in SeqIO.parse(args.fas, "fasta") if rec.description != ref_acc]
    total_sequences = len(tasks)
    logging.info(f"Loaded {total_sequences} query sequences.")

    logging.info("Pre-processing reference sequence for Numba...")
    refseq_int = INT_MAP_ARR[np.frombuffer(ref_seq_str.encode('ascii', 'ignore'), dtype=np.uint8)]

    os.makedirs(args.o, exist_ok=True)
    mutations_csv, stats_csv = os.path.join(args.o, "mutations_result.csv"), os.path.join(args.o, "mutations_stat.csv")
    
    stats_headers = ['Ref_ID', 'Ref_length', 'Query_ID', 'Query_length', 'SNP#', 'Insertion#', 'Deletion#', 'Indel#', 'Similarity']
    mutations_headers = ['Query_ID', 'Mutations']
    pd.DataFrame(columns=stats_headers).to_csv(stats_csv, index=False)
    pd.DataFrame(columns=mutations_headers).to_csv(mutations_csv, index=False)
    
    ref_stats_df = pd.DataFrame([{'Ref_ID': ref_acc, 'Ref_length': ref_length, 'Query_ID': ref_acc, 'Query_length': ref_length, 'SNP#': 0, 'Insertion#': 0, 'Deletion#': 0, 'Indel#': 0, 'Similarity': "100.00"}])
    ref_muts_df = pd.DataFrame([{'Query_ID': ref_acc, 'Mutations': ''}])
    ref_stats_df.to_csv(stats_csv, mode='a', header=False, index=False)
    ref_muts_df.to_csv(mutations_csv, mode='a', header=False, index=False)

    if total_sequences > 0:
        processes = min(args.t, total_sequences)
        logging.info(f"Starting analysis of {total_sequences} sequences with {processes} processes...")
        
        chunksize = max(1, total_sequences // (processes * 4))
        if chunksize > 4096: chunksize = 4096
        
        init_args = (refseq_int, ref_acc, ref_length)
        with mp.Pool(processes=processes, initializer=_init_worker, initargs=init_args) as pool:
            results_iter = pool.imap_unordered(process_sequence_worker, tasks, chunksize=chunksize)
            
            stats_buffer, muts_buffer, processed_count = [], [], 0
            for stats_res, muts_res in results_iter:
                stats_buffer.append(stats_res)
                muts_buffer.append(muts_res)
                if len(stats_buffer) >= WRITE_BUFFER_SIZE:
                    pd.DataFrame(stats_buffer).to_csv(stats_csv, mode='a', header=False, index=False)
                    pd.DataFrame(muts_buffer).to_csv(mutations_csv, mode='a', header=False, index=False)
                    stats_buffer.clear(); muts_buffer.clear()
                processed_count += 1
                if processed_count % 500 == 0 or processed_count == total_sequences:
                    logging.info(f"Progress: {processed_count}/{total_sequences} ({(processed_count/total_sequences*100):.1f}%)")

        if stats_buffer:
            pd.DataFrame(stats_buffer).to_csv(stats_csv, mode='a', header=False, index=False)
            pd.DataFrame(muts_buffer).to_csv(mutations_csv, mode='a', header=False, index=False)
    logging.info(f"Output files: {mutations_csv}, {stats_csv}")
    return 0

if __name__ == "__main__":
    if sys.platform != 'linux':
        mp.set_start_method('spawn', force=True)
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        logging.error(f"An unexpected error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)