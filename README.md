# TempSnap-Trace: Temporal H haplotype Network Analysis Pipeline

## Overview

TempSnap-Trace is a Python-based pipeline designed for the dynamic analysis of viral evolution using temporal haplotype networks. It processes raw sequence data and associated metadata, constructs time-resolved evolutionary networks based on McAN simulations, performs community detection using Infomap, extracts evolutionary backbones representing key transmission pathways, and tracks specific community evolution chains over time. The pipeline is modular, allowing users to run the entire workflow or specific steps independently. It leverages parallel processing (`multiprocessing`) and optionally GPU acceleration (via PyTorch for the `Trace` module) to handle large datasets efficiently.

The core workflow consists of the following steps, primarily orchestrated by `main.py` for steps 1-4 and utilizing the `Trace.py` module for step 5:

1.  **Raw Data Processing (`rawdata`):**
    *   Filters input sequences based on quality (N content via `--ratio`, optional minimum length).
    *   Performs multiple sequence alignment against a specified reference sequence using the external tool `halign4`.
    *   Identifies variants relative to the reference and generates a canonical mutation string (`mutation_str`) for each unique haplotype using the script `variant_mark_ljj.py`.
    *   Integrates variant information (including the `mutation_str`) with metadata (e.g., date, location, lineage) into a processed data table (`*_processed_data_*.csv`).
2.  **McAN Table Generation (`mcantables`):**
    *   Utilizes the processed data table (specifically the `mutation_str` and date information).
    *   Runs McAN simulations across specified time intervals (`--interval`) within the defined date range (`--start`, `--end`).
    *   Generates haplotype relationship data (ancestor-descendant links) for each time point, stored in an HDF5 file (`McAN_raw_results_*.h5`).
3.  **Temporal Network Construction (`networks`):**
    *   Builds a weighted, directed graph (`igraph.Graph` object) for each time snapshot based on the McAN simulation results.
    *   Nodes represent unique haplotypes, identified by their `name` attribute (which corresponds to the `mutation_str`).
    *   Edges represent ancestor-descendant relationships identified by McAN.
    *   Edge weights (`evo_weight`) are calculated as the Jaccard similarity between the `mutation_str` of connected nodes.
    *   Node attributes are added, including by default `Date`, `Location`, `ID`, `Ancestor_ID`, `Lineage`, `Clade` (if present in the processed data), plus any *additional* attributes specified via `--attrs`.
    *   Calculates node importance using WDKS (Weighted Degree k-shell), stored as the `wdks` node attribute.
    *   Graphs are stored in an HDF5 file (`Temporal_graphs_*.h5`).
4.  **Community Detection & Backbone Extraction (`community`):**
    *   Applies the Infomap algorithm (`g.community_infomap`) to detect community structures within each temporal graph, using `evo_weight` for edges and `wdks` for nodes.
    *   Identifies key nodes within each community (typically the node with the maximum WDKS).
    *   Extracts the evolutionary backbone network by finding shortest paths (weighted by inverse `evo_weight`) between key nodes within connected components.
    *   Generates HDF5 files detailing community memberships (`Community_structures_*.h5`), network metrics (`Network_metrics_*.h5`), backbone graphs (`Backbone_networks_*.h5`), and backbone node tables (`Backbone_tables_*.h5`).
5.  **Community Evolution Tracking (`Trace` module):**
    *   Utilizes the outputs from the `networks` and `community` steps (graphs and partitions).
    *   Identifies communities containing a specific `label_of_interest` (e.g., a target mutation string) within a designated `tracking_label` node attribute (e.g., 'name').
    *   Traces the predecessors and successors of these target communities across time steps based on haplotype similarity (calculated using CPU or GPU via PyTorch).
    *   Constructs evolution chains, represented as pandas DataFrames, detailing community properties, lineage compositions, and temporal links.
    *   Results are typically saved to an HDF5 file (`tracking_results_*.h5`).

## Prerequisites

### Software

*   **Python:** Version 3.13.4.
*   **External Tools:**
    *   **halign4:** Required for the `rawdata` step (Multiple Sequence Alignment). Must be installed and accessible in the system's PATH.
    *   **variant\_mark\_ljj.py:** Required by `Get_data.py` during the `rawdata` step. Must be present and executable by the Python interpreter running the pipeline.

### Python Dependencies

Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## Usage (`main.py` Pipeline - Steps 1-4)

The main pipeline steps (`rawdata`, `mcantables`, `networks`, `community`) are executed via the `main.py` script.

```bash
python main.py --command <step1> [<step2> ...] [options]
```

### Commands (`--command` for `main.py`)

Specify one or more pipeline steps to execute. Steps are generally run in the order listed below. If a step is run, its output is typically used as input for the subsequent step unless overridden by specific file path arguments (`--samples`, `--tables`, `--graphs`).

*   **`all` (Default):** Executes the complete pipeline sequentially: `rawdata` -> `mcantables` -> `networks` -> `community`.
*   **`rawdata`:** Performs Step 1: Raw Data Processing.
    *   *Requires:* `--input_dir`, `--output_dir`.
    *   *Dependencies:* `halign4`, `variant_mark_ljj.py`.
    *   *Outputs:* Processed samples CSV file (`*_processed_data_*.csv`), alignment file, variant files, quality report, log file.
*   **`mcantables`:** Performs Step 2: McAN Table Generation.
    *   *Requires:* Processed samples data (from `rawdata` step or via `--samples`), `--output_dir`, date range (`--start`/`--end` or auto-detected).
    *   *Dependencies:* `McAN` module (assumed provided/importable).
    *   *Outputs:* McAN results HDF5 file (`McAN_raw_results_*.h5`), log file (appended).
*   **`networks`:** Performs Step 3: Temporal Network Construction.
    *   *Requires:* McAN results (from `mcantables` step or via `--tables`), `--output_dir`.
    *   *Outputs:* Temporal graphs HDF5 file (`Temporal_graphs_*.h5`), log file (appended).
*   **`community`:** Performs Step 4: Community Detection & Backbone Extraction.
    *   *Requires:* Temporal graphs (from `networks` step or via `--graphs`), McAN results (from `mcantables` step or via `--tables`), `--output_dir`.
    *   *Outputs:* Community structures (`Community_structures_*.h5`), network metrics (`Network_metrics_*.h5`), backbone networks (`Backbone_networks_*.h5`), and backbone tables (`Backbone_tables_*.h5`) (all as HDF5 files), log file (appended).

### Arguments (`main.py`)

#### General & Input/Output Control
*   `--command <step1> [<step2> ...]`: Specifies the pipeline step(s) to run. Choices: `all`, `rawdata`, `mcantables`, `networks`, `community`. Default: `all`.
*   `--output_dir <DIR>`: **Required for all steps.** Directory where all generated output files (including `log.txt`) will be saved.
*   `--input_dir <DIR>`: **Required for `rawdata` step.** Directory containing input sequence files (FASTA format: `.fasta`, `.fa`, `.fna`) and metadata files (formats: `.tsv`, `.xlsx`, `.xls`, `.csv`). The script expects one sequence file and one metadata file in this directory.
*   `--samples <FILE>`: Path to the processed samples CSV file (tab-separated, output of `rawdata`). **Required for `mcantables` if `rawdata` is not run.** Overrides the output from the `rawdata` step if provided. Contains essential columns like `mutation_str`, `Date`, and metadata.
*   `--tables <FILE>`: Path to the HDF5 file containing McAN simulation results (list of DataFrames, output of `mcantables`). **Required for `networks` and `community` if `mcantables` is not run.** Overrides the output from the `mcantables` step if provided.
*   `--graphs <FILE>`: Path to the HDF5 file containing the list of temporal graphs (`igraph.Graph` objects, output of `networks`). **Required for `community` if `networks` is not run.** Overrides the output from the `networks` step if provided.
*   `--p <N>`: Number of processes/threads to use for parallelizable tasks (alignment, variant marking, McAN, graph building, community detection). Default: 4.

#### Raw Data Processing (`rawdata`) Specific
*   `--ratio <R>`: Maximum allowed ratio of 'N' bases in a sequence during filtering. Default: 0.001 (0.1%).
*   `--ref <ID>`: Reference sequence identifier (exact or partial) used for alignment (`halign4`) and variant calling (`variant_mark_ljj.py`). Default: 'EPI_ISL_402125'.
*   *(Note: A minimum sequence length filter (`min_size`) exists internally in `Get_data.py` but is not exposed as a command-line argument in `main.py` and defaults to 0.)*

#### Temporal Analysis (`mcantables`, `networks`, `community`) Specific
*   `--start <DATE>`: Start date for the temporal analysis window (format: YYYY-MM-DD). If not provided, the script attempts to auto-detect the minimum date from the 'Date' column in the processed samples data. **Required if auto-detection fails.**
*   `--end <DATE>`: End date for the temporal analysis window (format: YYYY-MM-DD). If not provided, the script attempts to auto-detect the maximum date from the 'Date' column in the processed samples data. **Required if auto-detection fails.**
*   `--interval <DAYS>`: Time interval in days used to create discrete time snapshots for McAN simulation and network construction. Default: 7.
*   `--attrs <ATTR> [<ATTR> ...]`: List of *additional* metadata attribute column names (beyond the defaults like Lineage, Clade, Location) from the processed data file to be included as node attributes in the temporal graphs. Default: [].

## Usage (`Trace` Module - Step 5: Community Tracking)

The community evolution tracking functionality is provided by the `Trace.py` module and is typically used *after* running the `networks` and `community` steps of the main pipeline, as it requires the generated graphs and partitions. It is invoked by calling the `track_community_evolution` function, usually within a Python script or Jupyter Notebook.

### `track_community_evolution` Function

This function identifies communities containing a specific `label_of_interest` within a given `tracking_label` node attribute. It then traces the predecessors and successors of these communities across time steps based on haplotype similarity, constructing evolution chains.

**Key Arguments:**

*   `partitions` (List): The list of community partitions (list of lists of node names/`mutation_str`) generated by the `community` step (or loaded from `Community_structures_*.h5`).
*   `extended_graphs` (List): The list of `igraph.Graph` objects generated by the `networks` step (or loaded from `Temporal_graphs_*.h5`).
*   `label_of_interest` (str): The specific value within the `tracking_label` attribute that identifies the target communities to start tracking from.
*   `tracking_label` (str): The node attribute key (e.g., 'name', 'ID') used to find the `label_of_interest`.
*   `recording_label` (str): The node attribute key (e.g., 'Lineage', 'Clade') used for counting types/lineages within communities for the output DataFrame.
*   `start_date` (Optional[str]): Start date ('YYYY-MM-DD') to filter the tracking window. If None, uses the earliest date in graphs.
*   `end_date` (Optional[str]): End date ('YYYY-MM-DD') to filter the tracking window. If None, uses the latest date in graphs.
*   `time_interval` (int): Assumed days between graph snapshots if dates are missing (used for date range estimation). Default: 7.
*   `similarity_threshold` (float): Minimum similarity score (0-1) required to consider two communities linked. Default: 0.4.
*   `weight_attr` (str): Node attribute used to identify the 'core' node of a community (based on max value). Default: "wdks".
*   `n_processes` (int): Number of CPU cores for parallel processing (primarily chain building). Default: 4.
*   `time_window` (int): How many previous time steps to search for potential predecessors. Default: 2.
*   `output_path` (Optional[str]): Path to save the resulting list of chain DataFrames in HDF5 format. If None, a default path is generated.

**Returns:**

*   A list of pandas DataFrames. Each DataFrame represents a unique evolution chain containing the `label_of_interest`, indexed by Date and showing community details, predecessors, similarity, and lineage counts. Returns an empty list if no chains are found or an error occurs.

## Input File Formats

*   **Sequence File (`--input_dir`):** Standard FASTA format (`.fasta`, `.fa`, `.fna`). Headers should ideally contain the sequence identifier and collection date (YYYY-MM-DD or YYYY-MM).
*   **Metadata File (`--input_dir`):** Tabular format (`.tsv`, `.csv`, `.xls`, `.xlsx`). Must contain columns for:
    *   Sequence Identifier (e.g., 'Accession ID', 'Isolate\_Id', 'gisaid\_epi\_isl')
    *   Collection Date (e.g., 'date', 'Collection\_Date') - YYYY-MM-DD or YYYY-MM format preferred.
    *   Optional: Lineage ('Lineage', 'Pango lineage'), Clade ('Clade', 'GISAID\_clade'), Location ('Location', 'country'). The script attempts case-insensitive matching for common column names during the `rawdata` step.

## Output Files

All output files are saved in the directory specified by `--output_dir` (for `main.py`) or potentially a custom path (`output_path` for `track_community_evolution`). Filenames often include the base name of the input FASTA file and potentially date ranges. A central log file `log.txt` is created/appended to in the `--output_dir` by `main.py`.

### `rawdata` Step Outputs
*   `*_filtered_N_lt_*.fasta`: Filtered FASTA sequences.
*   `*_quality_report_*.csv`: Report on sequence length and N content.
*   `*_halign4_*.fasta`: Multiple sequence alignment output from `halign4`.
*   `*_mutations_result.csv`: Output from `variant_mark_ljj.py` listing mutations per sequence.
*   `*_mutations_stats.csv`: Output from `variant_mark_ljj.py` with variant statistics per sequence.
*   `*_processed_data_*.csv`: **Key output.** Tab-separated file combining sequence IDs, the canonical `mutation_str`, and merged metadata. This file serves as the primary input for the `mcantables` step. Its main columns are:
    *   `ID`: The sequence identifier (e.g., Accession ID), extracted from the metadata.
    *   `Date`: The collection date (parsed into YYYY-MM-DD format, partial dates like YYYY-MM are treated as the first of the month).
    *   `Location`: Geographical location, extracted from metadata (attempts 'Location', falls back to 'Country'). Defaults to 'Unknown'.
    *   `Lineage`: Assigned lineage (e.g., Pango lineage), extracted from metadata. Defaults to 'Unknown'.
    *   `Clade`: Assigned clade (e.g., GISAID or Nextstrain clade), extracted from metadata. Defaults to 'Unknown'.
    *   `Mutations_str`: A semicolon-separated string representing the differences between the sequence and the reference, generated by `variant_mark_ljj.py`. The format for each mutation is:
        *   **SNP:** `RefPos(SNP:RefBase->QueryBase)` (e.g., `23403(SNP:A->G)`)
        *   **Insertion:** `RefPosAfter(Insertion:BaseBefore->BaseBeforeInsertedBases)` (e.g., `11288(Insertion:C->CT)`) - `RefPosAfter` is the 1-based position in the reference *after* the base preceding the insertion.
        *   **Deletion:** `RefPosAfter(Deletion:BaseBeforeDeletedBases->BaseBefore)` (e.g., `21990(Deletion:CAT->C)`) - `RefPosAfter` is the 1-based position in the reference *after* the base preceding the deletion.
        *   **Indel:** Similar format to Insertion/Deletion but marked as `Indel` if the base preceding the indel also differs (this specific classification might depend on `variant_mark_ljj.py`'s exact logic).
        *   Example combined string: `3037(SNP:C->T);14408(SNP:C->T);23403(SNP:A->G)`

### `mcantables` Step Outputs
*   `McAN_raw_results_YYYY-MM-DD_to_YYYY-MM-DD.h5`: HDF5 file storing a list of pandas DataFrames (one per snapshot) representing the raw McAN simulation results (ancestor-descendant relationships). Uses Blosc2 compression.

### `networks` Step Outputs
*   `Temporal_graphs_YYYY-MM-DD_to_YYYY-MM-DD.h5`: HDF5 file storing a list of `igraph.Graph` objects (one per snapshot). Nodes are haplotypes (`name` = `mutation_str`), edges are weighted by `evo_weight` (Jaccard similarity), nodes have metadata and `wdks` attributes. Uses Blosc2 compression.

### `community` Step Outputs
*   `Community_structures_YYYY-MM-DD_to_YYYY-MM-DD.h5`: HDF5 file storing a list of community partitions (list of lists of node names/`mutation_str`) for each snapshot.
*   `Network_metrics_YYYY-MM-DD_to_YYYY-MM-DD.h5`: HDF5 file storing a list of tuples `(modularity, codelength)` for each snapshot's partitioning.
*   `Backbone_networks_YYYY-MM-DD_to_YYYY-MM-DD.h5`: HDF5 file storing a list of `igraph.Graph` objects representing the extracted backbone network for each snapshot.
*   `Backbone_tables_YYYY-MM-DD_to_YYYY-MM-DD.h5`: HDF5 file storing a list of pandas DataFrames with details (metadata, `wdks`, `is_key_node`) about the nodes included in each backbone network snapshot.

### `track_community_evolution` Function Output
*   `tracking_results_*.h5` (or custom name via `output_path`): HDF5 file storing a list of pandas DataFrames, each representing a unique evolution chain.

## Examples

### `main.py` Pipeline Examples

1.  **Run Full Pipeline (Steps 1-4, using most arguments):**
    ```bash
    python main.py \
        --command all \
        --input_dir /home/liujiajun/projects/Hap_networks/module/South_Africa/data/South_Africa_Beta \
        --output_dir /home/liujiajun/projects/Hap_networks/module/South_Africa/result_0421 \
        --start 2020-03-12 \
        --end 2020-12-17 \
        --interval 7 \
        --ref EPI_ISL_402125
    ```
    *(Runs all steps sequentially, specifying input/output, filtering, reference, processes, date range, interval, and additional attributes.)*

2.  **Run Raw Data Processing Step Only (Step 1):**
    ```bash
    python main.py \
        --command rawdata \
        --input_dir /home/liujiajun/projects/Hap_networks/data/South_Africa_Beta \
        --output_dir /home/liujiajun/projects/Hap_networks/module/South_Africa/results \
        --start 2020-03-12 \
        --end 2020-12-17 \
        --interval 7
    ```

3.  **Run McAN Table Generation Step Only (Step 2):**
    ```bash
    python main.py \
        --command mcantables \
        --samples /home/liujiajun/projects/Hap_networks/module/South_Africa/results/gisaid_hcov-19_2025_03_29_12_processed_data_2020_03_06-2020_12_10.csv \
        --output_dir /home/liujiajun/projects/Hap_networks/module/South_Africa/results \
        --start 2020-03-12 \
        --end 2020-12-10 \
        --interval 7
    ```

4.  **Run Network Construction Step Only (Step 3):**
    ```bash
    python main.py \
        --command networks \
        --tables /home/liujiajun/projects/Hap_networks/module/South_Africa/results/McAN_raw_tables_2020-03-12-2020-12-10.h5 \
        --output_dir /home/liujiajun/projects/Hap_networks/module/South_Africa/results \
        --start 2020-03-12 \
        --end 2020-12-10 \
        --interval 7
    ```

5.  **Run Community Detection Step Only (Step 4):**
    ```bash
    python main.py \
        --command community \
        --graphs /home/liujiajun/projects/Hap_networks/module/South_Africa/results/networks_2020-03-12-2020-12-10.h5 \
        --output_dir /home/liujiajun/projects/Hap_networks/module/South_Africa/results \
        --start 2020-03-12 \
        --end 2020-12-10 \
        --interval 7
    ```

### `Trace` Module Example (Step 5 - Jupyter Notebook)

```python
# Assuming you are in a Jupyter Notebook (.ipynb)
# and have already run steps 1-4 or loaded the required HDF5 files

import pandas as pd
from TempSnap import IOManager # Assuming IOManager is used for loading
from Trace import track_community_evolution
import igraph as ig # Required if loading graphs manually

# --- Load required data (Output from previous steps) ---
print("Loading data files...")

# Load McAN tables
mcan_tables = IOManager.load_from_hdf5(
    '/home/liujiajun/projects/Hap_networks/module/Global_dataset/results/McAN_raw_results_2020-03-25_to_2021-03-24.h5',
    parallelism_level=12
)

# Load temporal graphs
graphs = IOManager.load_from_hdf5(
    '/home/liujiajun/projects/Hap_networks/module/04-15/0605/Global_results_new/Temporal_graphs_2020-03-25_to_2021-03-24.h5',
    parallelism_level=12
)

# Load community structures
communities = IOManager.load_from_hdf5(
    '/home/liujiajun/projects/Hap_networks/module/04-15/0605/Global_results_new/Community_structures_2020-03-25_to_2021-03-24.h5'
)

# Load backbone networks
backbone_graphs = IOManager.load_from_hdf5(
    '/home/liujiajun/projects/Hap_networks/module/Global_dataset/results/Backbone_networks_2020-03-25_to_2021-03-24.h5'
)

# Load backbone tables
backbone_dfs = IOManager.load_from_hdf5(
    '/home/liujiajun/projects/Hap_networks/module/Global_dataset/results/Backbone_tables_2020-03-25_to_2021-03-24.h5'
)

# Load network metrics evolution
metrics_evo = IOManager.load_from_hdf5(
    '/home/liujiajun/projects/Hap_networks/module/Global_dataset/results/Network_metrics_2020-03-25_to_2021-03-24.h5'
)

print(f"Successfully loaded data files:")
print(f"- Communities: {len(communities) if communities else 0} time points")
print(f"- Graphs: {len(graphs) if graphs else 0} time points")

# --- Run Community Evolution Tracking (Step 5) ---
if communities and graphs:
    print("Starting community evolution tracking...")
    
    tracking_chains_fran3 = track_community_evolution(
        partitions=communities,
        extended_graphs=graphs,
        label_of_interest='3037(SNP:C->T);14408(SNP:C->T);15324(SNP:C->T);23403(SNP:A->G)',
        tracking_label='name',
        recording_label='Lineage',
        start_date='2020-03-25',
        end_date='2020-11-04',
        time_interval=7,
        similarity_threshold=0.4,
        output_path='/home/liujiajun/projects/Hap_networks/module/04-15/0605/B_1_1_tracking_chains_Key_France.h5'
    )

    # --- Process Results ---
    if tracking_chains_fran3:
        print(f"Successfully generated {len(tracking_chains_fran3)} unique evolution chains.")
        print("Results saved to specified output path.")
        
        # Display the first chain as an example
        print("\nExample Chain (First DataFrame):")
        display(tracking_chains_fran3[0]) # Use display() in Jupyter for nice formatting
    else:
        print("Tracking did not produce any valid chains for the specified label and parameters.")
else:
    print("Input data (communities or graphs) not loaded or invalid. Cannot run tracking.")

```

## Notes

*   **File Paths:** The pipeline relies on specific naming conventions for intermediate files when running sequential steps (`all` or multiple commands via `main.py`). When running steps individually or using the `Trace` module, ensure the required input file paths (`--samples`, `--tables`, `--graphs`, or paths loaded in scripts) are correctly specified.
*   **Date Handling:** Date parsing relies on common formats (YYYY-MM-DD, YYYY-MM). Inconsistent metadata dates can cause errors. Auto-detection in `main.py` requires a 'Date' column in the processed data.
*   **Memory Usage:** Can be significant, especially for `networks`, `community`, and `Trace` steps with large datasets or many time points. Adjust `--p` (or `n_processes`) based on available RAM and CPU cores.
*   **Parallel Processing:** Uses `multiprocessing` with the `spawn` context by default for better cross-platform compatibility, creating new processes for parallel tasks.
*   **GPU Usage (Trace):** Requires PyTorch and compatible NVIDIA drivers/CUDA toolkit for GPU acceleration in `track_community_evolution`. Falls back to CPU if unavailable or `use_gpu=False`.
*   **HDF5 Storage:** Intermediate and final results (McAN tables, graphs, community results, tracking chains) are stored in HDF5 format (`.h5`) using `h5py`, `joblib`, and `blosc2` compression for efficiency. Ensure sufficient disk space.
*   **Logging:** `main.py` logs progress and errors to `stdout` and detailed information to `log.txt` within the `--output_dir`. The `Trace` module uses standard Python logging.

## Citation

If you use this pipeline in your research, please cite:

*[Insert appropriate citation details here - e.g., associated publication, software DOI, repository link]*
