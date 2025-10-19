# Author: Liu Jiajun
# Date: 2025-10-19
#
# This script is written by Liu Jiajun. The corresponding article is "TempSnap-Trace: A Temporal Snapshot-Based Framework for Haplotype Network Tracing."
import os
import argparse
import Get_data
import TempSnap
from typing import Optional
import pandas as pd
import logging

def main():
    """Main function that executes different workflows based on command parameter"""

    # Create parser
    parser = argparse.ArgumentParser(description='Viral Genomics Analysis Pipeline')

    parser.add_argument('--command', nargs='*', default=['all'],
                        choices=['all', 'rawdata', 'mcantables', 'networks', 'community'],
                        help='Command(s) to execute: all (full pipeline), rawdata, mcantables, networks, community. Specify multiple commands separated by spaces. (default: all)')

    parser.add_argument('--input_dir', metavar='PATH', help='Input directory OR a single FASTA file (file => embedded) (required for rawdata)')
    parser.add_argument('--output_dir', metavar='DIR', help='Output directory (required for all steps)')
    parser.add_argument('--ratio', metavar='R', type=float, default=0.001, help='Maximum N base ratio (default: 0.001)')
    parser.add_argument('--ref', metavar='ID', default='EPI_ISL_402125', help='Reference sequence name')
    parser.add_argument('--p', dest='n', metavar='N', type=int, default=4, help='Number of processes (default: 4)')

    parser.add_argument('--samples', dest='samples_path', metavar='FILE', help='Path to processed samples CSV file (required for mcantables if rawdata not run)') # Renamed dest
    parser.add_argument('--start', dest='start_date', metavar='DATE', help='Start date (YYYY-MM-DD) (required for mcantables/networks/community if not detectable)')
    parser.add_argument('--end', dest='end_date', metavar='DATE', help='End date (YYYY-MM-DD) (required for mcantables/networks/community if not detectable)')
    parser.add_argument('--interval', dest='time_interval', metavar='DAYS', type=int, default=7,
                        help='Time interval in days (default: 7)')
    parser.add_argument('--attrs', dest='optional_attrs', metavar='ATTR', nargs='+', default=[],
                        help='Optional attributes for analysis')
    parser.add_argument('--tables', dest='raw_results_path', metavar='FILE', # Renamed dest
                        help='Path to raw McAN results file (.h5) (required for networks/community if mcantables not run)')
    parser.add_argument('--graphs', dest='graphs_path', metavar='FILE', # Renamed dest
                        help='Path to temporal graphs file (.h5) (required for community if networks not run)')
    args, _ = parser.parse_known_args()  # Use known_args to avoid errors on unknown args
    
    # Configure logging for the main process
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        TempSnap.LogManager.configure_logging(args.output_dir, process_type="main")
    else:
        logging.warning("No output directory specified, logs will only be printed to console.")
    args = parser.parse_args()

    # Determine which commands to run
    commands_to_run = args.command
    run_all = 'all' in commands_to_run
    requested_set = set(commands_to_run)

    # Define the logical order of pipeline steps
    pipeline_steps = ['rawdata', 'mcantables', 'networks', 'community']

    # Variables to store intermediate results/paths
    df = None
    # Initialize intermediate file paths with arguments, they might be overwritten if steps are run
    processed_data_file = args.samples_path # Use renamed arg
    raw_results_file = args.raw_results_path # Use renamed arg
    graphs_path = args.graphs_path         # Use renamed arg
    tempsnap_shared: Optional[TempSnap.TempSnap] = None
    # Initialize dates with arguments
    start_date = args.start_date
    end_date = args.end_date

    logging.info(f"Requested commands: {commands_to_run}")
    if run_all:
        logging.info("Running full pipeline ('all' specified).")

    # --- Step 1: Raw Data Processing ---
    if run_all or 'rawdata' in requested_set:
        logging.info("===== STEP 1: RAW DATA PROCESSING =====")
        if not args.input_dir or not args.output_dir:
            parser.error("--input_dir and --output_dir are required for 'rawdata' step")

        try:
            df_step1, processed_data_file_step1 = Get_data.run_pipeline(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                ratio=args.ratio,
                ref=args.ref,
                n=args.n
            )

            if df_step1 is None or processed_data_file_step1 is None or not os.path.exists(processed_data_file_step1):
                logging.error("Raw data processing failed or did not produce the expected output file.")
                return

            # Update variables for subsequent steps
            df = df_step1
            processed_data_file = processed_data_file_step1 # Overwrite if run

            # Determine date range if not provided by args (only if rawdata was run)
            if not start_date or not end_date:
                logging.info("Attempting to detect date range from processed data...")
                try:
                    if df is not None and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        dates = df['Date'].dropna()
                        if not dates.empty:
                            detected_start = dates.min().strftime('%Y-%m-%d')
                            detected_end = dates.max().strftime('%Y-%m-%d')
                            logging.info(f"Detected date range: {detected_start} to {detected_end}")
                            if not start_date: start_date = detected_start
                            if not end_date: end_date = detected_end
                        else: logging.warning("'Date' column empty or invalid.")
                    else: logging.warning("Cannot detect dates (DataFrame missing or no 'Date' column).")
                except Exception as e: logging.warning(f"Error detecting dates: {e}")

        except Exception as e:
             logging.error(f"Error during Raw Data Processing: {e}")
             return

    # --- Step 2: Generate McAN Tables (now McAN Results) ---
    if run_all or 'mcantables' in requested_set:
        logging.info("===== STEP 2: GENERATING McAN RESULTS =====")
        # Check required inputs for this step
        if not processed_data_file or not os.path.exists(processed_data_file):
             parser.error("Input samples file not found. Provide --samples or run 'rawdata' first.")
        if not args.output_dir:
             parser.error("--output_dir is required for 'mcantables' step.")

        # Ensure dates are available
        if not start_date or not end_date:
             # Try to determine dates if not already done (e.g., if only mcantables is run)
             if df is None and processed_data_file and os.path.exists(processed_data_file):
                 logging.info("Attempting to read dates from provided samples file...")
                 try:
                     # Read only necessary columns to save memory
                     temp_df = pd.read_csv(processed_data_file, sep='\t', usecols=['Date'], low_memory=False)
                     temp_df['Date'] = pd.to_datetime(temp_df['Date'], errors='coerce')
                     dates = temp_df['Date'].dropna()
                     if not dates.empty:
                         if not start_date: start_date = dates.min().strftime('%Y-%m-%d')
                         if not end_date: end_date = dates.max().strftime('%Y-%m-%d')
                         logging.info(f"Using date range from samples file: {start_date} to {end_date}")
                     else: logging.warning("Could not determine date range from samples file ('Date' column empty or invalid).")
                 except Exception as e: logging.warning(f"Error reading dates from samples file: {e}")

             if not start_date or not end_date: # Final check
                 parser.error("Could not determine date range for McAN results. Please provide --start and --end.")

        logging.info(f"Using date range for McAN results: {start_date} to {end_date}")

        try:
            tempsnap_shared = TempSnap.TempSnap(
                output_path=args.output_dir,
                samples_path=processed_data_file,
                start_date=start_date,
                end_date=end_date,
                time_interval=args.time_interval,
                num_processes=args.n,
                optional_attrs=args.optional_attrs
            )
            raw_results_file_step2 = tempsnap_shared.run_mcan_simulation()
            if not raw_results_file_step2 or not os.path.exists(raw_results_file_step2):
                logging.error("McAN results generation failed or did not produce output file.")
                return
            raw_results_file = raw_results_file_step2
        except Exception as e:
             logging.error(f"Error during McAN results generation: {e}")
             return

    # --- Step 3: Build Temporal Graphs ---
    if run_all or 'networks' in requested_set:
        logging.info("===== STEP 3: BUILDING TEMPORAL GRAPHS =====")
        raw_data_in_memory = tempsnap_shared and tempsnap_shared.raw_networks_data
        raw_data_on_disk = raw_results_file and os.path.exists(raw_results_file)
        if not raw_data_in_memory and not raw_data_on_disk:
            parser.error("Input McAN results not available. Provide --tables or run 'mcantables' first.")
        if not args.output_dir:
            parser.error("--output_dir is required for 'networks' step.")

        try:
            # Create a TempSnap instance for this step
            tempsnap_net = tempsnap_shared or TempSnap.TempSnap(
                output_path=args.output_dir,
                samples_path=processed_data_file,
                start_date=start_date,
                end_date=end_date,
                time_interval=args.time_interval,
                num_processes=args.n,
                optional_attrs=args.optional_attrs
            )
            tempsnap_shared = tempsnap_net
            graphs_path_step3 = tempsnap_net.build_temporal_graphs(
                raw_results_path=raw_results_file if raw_data_on_disk else None
            )
            if not graphs_path_step3 or not os.path.exists(graphs_path_step3):
                logging.error("Temporal graph construction failed or did not produce output file.")
                return
            graphs_path = graphs_path_step3
        except Exception as e:
             logging.error(f"Error during temporal graph construction: {e}")
             return

    # --- Step 4: Detect Communities ---
    if run_all or 'community' in requested_set:
        logging.info("===== STEP 4: DETECTING COMMUNITY STRUCTURE =====")
        graphs_in_memory = tempsnap_shared and tempsnap_shared.temporal_graphs
        graphs_on_disk = graphs_path and os.path.exists(graphs_path)
        if not graphs_in_memory and not graphs_on_disk:
            parser.error("Input graphs not available. Provide --graphs or run 'networks' first.")
        raw_data_in_memory = tempsnap_shared and tempsnap_shared.raw_networks_data
        raw_data_on_disk = raw_results_file and os.path.exists(raw_results_file)
        if not raw_data_in_memory and not raw_data_on_disk:
            parser.error("Input McAN results not available. Provide --tables or run 'mcantables' first.")
        if not args.output_dir:
            parser.error("--output_dir is required for 'community' step.")

        try:
            # Create a TempSnap instance for this step
            tempsnap_comm = tempsnap_shared or TempSnap.TempSnap(
                output_path=args.output_dir,
                samples_path=processed_data_file,
                start_date=start_date,
                end_date=end_date,
                time_interval=args.time_interval,
                num_processes=args.n,
                optional_attrs=args.optional_attrs
            )
            tempsnap_shared = tempsnap_comm
            comm_path, met_path, bb_path, bbt_path = tempsnap_comm.detect_communities_and_backbones(
                graphs_path=graphs_path if graphs_on_disk else None,
                raw_results_path=raw_results_file if raw_data_on_disk else None
            )
            # Optionally print the paths to the output files
        except Exception as e:
             logging.error(f"Error during community detection: {e}")
             return

    logging.info("Requested pipeline steps completed successfully!")

if __name__ == "__main__":
    main()