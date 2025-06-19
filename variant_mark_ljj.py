#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# variant_mark_ljj.py - Simplified multi-process optimized Python variant marking script

import argparse
import os
import re
import sys
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from functools import partial
from typing import Dict, List, Tuple, Optional, Any

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Variant marking tool")
    parser.add_argument("-fas", required=True, help="Input file, multiple aligned sequences (FASTA)")
    parser.add_argument("-ref", required=True, help="Input string, reference sequence ID (can be partial)")
    parser.add_argument("-o", required=True, help="Output directory")
    parser.add_argument("-t", type=int, default=mp.cpu_count(), help=f"Number of processes to use (default: {mp.cpu_count()})")
    # --- Base name argument is kept as it's essential for output naming ---
    parser.add_argument("--base_name", required=True, help="Base name for output files (e.g., derived from input fasta)")
    
    return parser.parse_args()

def read_aligned_sequences(fas_file: str) -> Dict[str, str]:
    """Efficiently read FASTA format multiple alignment sequence file"""
    aln_seq: Dict[str, List[str]] = {}
    name: Optional[str] = None
    
    try:
        with open(fas_file, 'r') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    name = line[1:]  # Remove '>' prefix
                    if name: # Ensure name is not empty
                        aln_seq[name] = []
                    else:
                        name = None # Invalid header
                elif name is not None:
                    aln_seq[name].append(line)
    except FileNotFoundError:
        print(f"ERROR: Input FASTA file not found: {fas_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read FASTA file {fas_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge sequence fragments and return final dictionary
    return {name: ''.join(parts) for name, parts in aln_seq.items()}

def find_reference_sequence(ref_acc: str, aln_seq: Dict[str, str]) -> Optional[str]:
    """
    Find the reference sequence using exact or partial ID matching.
    """
    # Try direct match first
    if ref_acc in aln_seq:
        print(f"Found exact match for reference ID: '{ref_acc}'")
        return ref_acc
    
    # Try partial match
    matching_keys = [key for key in aln_seq if ref_acc in key]
    
    if len(matching_keys) == 1:
        found_key = matching_keys[0]
        print(f"Found unique partial match for reference ID '{ref_acc}': '{found_key}'")
        return found_key
    elif len(matching_keys) > 1:
        found_key = matching_keys[0]
        print(f"WARNING: Multiple sequences match partial ID '{ref_acc}'. Using first match: '{found_key}'.", file=sys.stderr)
        print("First 5 matching sequences:", file=sys.stderr)
        for key in matching_keys[:5]:
            print(f"  '{key}'", file=sys.stderr)
        if len(matching_keys) > 5:
            print(f"  ...and {len(matching_keys) - 5} more.", file=sys.stderr)
        return found_key
    else:
        print(f"ERROR: No reference sequence matching ID '{ref_acc}' was found in the alignment.", file=sys.stderr)
        return None
    
def process_sequence(key: str, refseq_list: List[str], ref_acc: str, ref_length: int, aln_seq: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Analyze variants of a single sequence compared to the reference."""
    # Skip reference sequence itself
    if key == ref_acc:
        return None, None
        
    acgt_pattern = re.compile(r'[ACGT]', re.IGNORECASE)
    
    stats = {"SNP": 0, "Insertion": 0, "Deletion": 0, "Indel": 0}
    ref_pos = 0  # Reference sequence position (1-based)
    query_pos = 0  # Query sequence position (1-based, ungapped)
    
    query_seq_list = list(aln_seq[key])
    ref_seq_list_copy = refseq_list.copy() # Work on a copy
    mutation_details = []
    
    query_seq_ungapped = aln_seq[key].replace('-', '')
    query_length = len(query_seq_ungapped)

    # --- Sequence Cleaning Step ---
    # This part handles alignment artifacts where N/X align with gaps.
    # It's complex but aims to match the original logic. Simplification risks altering results.
    i = 0
    while i < len(ref_seq_list_copy) and i < len(query_seq_list):
        ref_char = ref_seq_list_copy[i]
        query_char = query_seq_list[i]
        
        # Original logic: if ref is '-' and query is empty string (impossible here), return None.
        # If ref is '-' and query is 'N' or 'X', remove both.
        if ref_char == "-" and query_char.upper() in ('N', 'X'):
            ref_seq_list_copy.pop(i)
            query_seq_list.pop(i)
            # Decrement i because the list length decreased and we need to re-check the new character at index i
            i -= 1 
        i += 1
    # --- End Cleaning Step ---

    # --- Variant Identification Loop ---
    i = 0
    while i < len(ref_seq_list_copy) and i < len(query_seq_list):
        ref_char = ref_seq_list_copy[i]
        query_char = query_seq_list[i]

        # Update reference position (1-based)
        if acgt_pattern.match(ref_char):
            ref_pos += 1
            
        # Update query position (1-based, ungapped)
        if query_char != '-':
            query_pos += 1
            
        # Replace N/X in query with reference base for comparison
        if query_char.upper() in ('N', 'X'):
            query_char = ref_char # Treat N/X as matching reference for variant calling
            query_seq_list[i] = ref_char # Update list for consistency if needed later (though not strictly necessary here)
            
        # --- Compare characters ---
        if ref_char != query_char:
            variant_type = "SNP" # Default assumption
            
            if ref_char == '-':  # Insertion in query relative to reference
                variant_type = "Insertion"
                start_ref_pos = ref_pos # Position *before* the insertion
                
                # Determine alleles involved (simplified logic)
                ref_allele_ins = ref_seq_list_copy[i-1] if i > 0 else ref_seq_list_copy[0] # Base before insertion in ref
                inserted_bases = ""
                
                # Find extent of insertion
                j = i
                ins_len = 0
                while j < len(ref_seq_list_copy) and ref_seq_list_copy[j] == '-':
                    inserted_bases += query_seq_list[j]
                    ins_len += 1
                    j += 1
                
                # Check if it's an Indel (base before insertion differs) - simplified check
                if i > 0 and ref_seq_list_copy[i-1] != query_seq_list[i-1]:
                     variant_type = "Indel"

                mutation_details.append(f"{start_ref_pos+1}({variant_type}:{ref_allele_ins}->{ref_allele_ins}{inserted_bases})")
                stats[variant_type] += 1
                i += ins_len - 1 # Advance main loop counter past the insertion gap in ref
                query_pos += ins_len -1 # Adjust query position count (already incremented once for the first base)

            elif query_char == '-':  # Deletion in query relative to reference
                variant_type = "Deletion"
                start_ref_pos = ref_pos -1 # Position *before* the deletion starts
                
                deleted_bases = ""
                ref_allele_del = ref_seq_list_copy[i-1] if i > 0 else '' # Base before deletion in ref

                # Find extent of deletion
                j = i
                del_len = 0
                while j < len(query_seq_list) and query_seq_list[j] == '-':
                    deleted_bases += ref_seq_list_copy[j]
                    del_len += 1
                    j += 1
                
                # Check if it's an Indel (base before deletion differs) - simplified check
                if i > 0 and ref_seq_list_copy[i-1] != query_seq_list[i-1]:
                     variant_type = "Indel"

                mutation_details.append(f"{start_ref_pos+1}({variant_type}:{ref_allele_del}{deleted_bases}->{ref_allele_del})")
                stats[variant_type] += 1
                i += del_len - 1 # Advance main loop counter past the deletion gap in query
                ref_pos += del_len -1 # Adjust ref position count (already incremented once for the first base)
                
            else:  # SNP
                # Original logic had a lookahead check related to adjacent indels.
                # Keep this for consistency, although its exact purpose might be subtle.
                k = i + 1
                is_complex_snp = False
                if k < len(ref_seq_list_copy) and k < len(query_seq_list):
                    if ref_seq_list_copy[k] == '-' or query_seq_list[k] == '-':
                        is_complex_snp = True # Part of a complex region, original code skipped detailed recording sometimes
                
                # Record SNP detail unless it's part of the complex adjacent indel case
                if not is_complex_snp:
                    mutation_details.append(f"{ref_pos}({variant_type}:{ref_char}->{query_char})")
                
                # Always count the SNP stat based on original logic
                stats[variant_type] += 1
                    
        i += 1
    # --- End Variant Identification Loop ---
    
    # Calculate similarity (Kimura 2-parameter based distance approximation?)
    total_vars = sum(stats.values())
    denominator = ref_length + query_length
    similarity = (1 - 2 * total_vars / denominator) * 100 if denominator > 0 else (100.0 if total_vars == 0 else 0.0)
    
    # Prepare return data
    stats_result = {
        'Ref_ID': ref_acc,
        'Ref_length': ref_length,
        'Query_ID': key,
        'Query_length': query_length,
        'SNP#': stats['SNP'],
        'Insertion#': stats['Insertion'],
        'Deletion#': stats['Deletion'], 
        'Indel#': stats['Indel'],
        'Similarity': f"{similarity:.2f}" # Format as string with 2 decimal places
    }
    
    mutation_result = {
        'Query_ID': key,
        'Mutations': ';'.join(mutation_details)
    }
    
    return stats_result, mutation_result

def calculate_chunksize(total_items: int, processes: int) -> int:
    """Calculate chunksize for multiprocessing pool based on total items and process count."""
    if processes <= 0:
        processes = 1
        
    # Basic calculation: aim for roughly 4-8 chunks per process
    chunksize = max(1, (total_items + processes * 4 - 1) // (processes * 4))
    
    # Apply some bounds to prevent excessively small or large chunks
    min_chunk = 10
    max_chunk = 1000 # Adjust if memory per task is very high/low
    
    return max(min_chunk, min(chunksize, max_chunk))

def main() -> int:
    """Main execution function"""
    start_time = datetime.now()
    args = parse_arguments()
    
    print(f"Reading sequences from: {args.fas}")
    aln_seq = read_aligned_sequences(args.fas)
    if not aln_seq:
        print("Error: No sequences read from alignment file.", file=sys.stderr)
        return 1 # Indicate error
    print(f"Read {len(aln_seq)} sequences.")
    
    print(f"Finding reference sequence matching: '{args.ref}'")
    ref_acc = find_reference_sequence(args.ref, aln_seq)
    if ref_acc is None:
        # Error message already printed by find_reference_sequence
        return 1 # Indicate error

    # Prepare reference sequence data (list for processing, string for length)
    refseq_list = list(aln_seq[ref_acc])
    refseq_ungapped = aln_seq[ref_acc].replace('-', '')
    ref_length = len(refseq_ungapped)
    print(f"Reference sequence '{ref_acc}' found, length (ungapped): {ref_length}")
    
    # Prepare output file paths
    try:
        os.makedirs(args.o, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Cannot create output directory '{args.o}': {e}", file=sys.stderr)
        return 1
        
    mutations_csv = os.path.join(args.o, f"{args.base_name}_mutations_result.csv")
    stats_csv = os.path.join(args.o, f"{args.base_name}_mutations_stats.csv")
    
    # Prepare sequence keys to process (excluding reference)
    sequence_keys = [key for key in aln_seq.keys() if key != ref_acc]
    total_sequences_to_process = len(sequence_keys)
    
    if total_sequences_to_process == 0:
        print("Warning: No sequences to process other than the reference sequence.", file=sys.stderr)
        # Still create empty output files for consistency? Or just exit?
        # Let's create files with only the reference row.

    # Initialize result lists and add reference data
    mutations_results: List[Dict[str, Any]] = [{'Query_ID': ref_acc, 'Mutations': ''}]
    stats_results: List[Dict[str, Any]] = [{
        'Ref_ID': ref_acc, 'Ref_length': ref_length, 'Query_ID': ref_acc,
        'Query_length': ref_length, 'SNP#': 0, 'Insertion#': 0,
        'Deletion#': 0, 'Indel#': 0, 'Similarity': "100.00"
    }]
    
    # Start multiprocessing if there are sequences to process
    if total_sequences_to_process > 0:
        print(f"Starting analysis of {total_sequences_to_process} sequences using {args.t} processes...")
        
        # Use partial to pre-fill arguments for the worker function
        process_func_partial = partial(
            process_sequence, 
            refseq_list=refseq_list, 
            ref_acc=ref_acc, 
            ref_length=ref_length, 
            aln_seq=aln_seq
        )
        
        chunksize = calculate_chunksize(total_sequences_to_process, args.t)
        print(f"Calculated chunk size: {chunksize}")

        try:
            with mp.Pool(processes=args.t) as pool:
                # Use imap_unordered for potentially better performance as results are processed as they complete
                results_iter = pool.imap_unordered(process_func_partial, sequence_keys, chunksize=chunksize)

                processed_count = 0
                start_process_time = datetime.now()
                stats_results = []
                mutations_results = []

                progress_intervals = [int(total_sequences_to_process * i * 0.2) for i in range(1, 6)]

                # Process results as they become available
                for result in results_iter:
                    stats, mutations = result
                    if stats is not None and mutations is not None:
                        stats_results.append(stats)
                        mutations_results.append(mutations)

                    processed_count += 1

                    if processed_count in progress_intervals or processed_count == total_sequences_to_process:
                        progress_percentage = (processed_count / total_sequences_to_process) * 100
                        sys.stderr.write(f"Mutation analysis progress: {progress_percentage:.0f}% ({processed_count}/{total_sequences_to_process})\n")
                        sys.stderr.flush()  # Ensure immediate output

        except Exception as e:
            print(f"\nERROR during multiprocessing: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Save results to CSV files
    print("Saving results...")
    try:
        mutations_df = pd.DataFrame(mutations_results)
        # Ensure consistent column order
        mutations_df = mutations_df[['Query_ID', 'Mutations']] 
        mutations_df.to_csv(mutations_csv, index=False)
        
        stats_df = pd.DataFrame(stats_results)
        # Ensure consistent column order
        stats_df = stats_df[[
            'Ref_ID', 'Ref_length', 'Query_ID', 'Query_length', 
            'SNP#', 'Insertion#', 'Deletion#', 'Indel#', 'Similarity'
        ]]
        stats_df.to_csv(stats_csv, index=False)
    except Exception as e:
        print(f"ERROR saving results to CSV: {e}", file=sys.stderr)
        return 1

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    print("\nAnalysis complete!")
    print(f"Total time: {elapsed_time.total_seconds():.2f} seconds")
    print(f"Output mutations: {mutations_csv}")
    print(f"Output statistics: {stats_csv}")
    return 0

if __name__ == "__main__":
    # Set start method to 'spawn' for better compatibility across platforms (macOS, Windows)
    # Needs to be done before creating the Pool
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError as e:
         # Might happen if context is already set, especially in interactive environments
         print(f"Note: Could not force multiprocessing start method to 'spawn': {e}", file=sys.stderr)
         pass 
    except AttributeError:
         # Handle systems where set_start_method might not be available (older Python?)
         print("Warning: multiprocessing.set_start_method not available.", file=sys.stderr)
         pass

    exit_code = 1 # Default to error
    try:
        exit_code = main()
    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred: {e}", file=sys.stderr)

        
    sys.exit(exit_code)