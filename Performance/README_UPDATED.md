# TempSnap vs VENAS Performance Comparison (Updated)

## Test Configuration

- **TempSnap Input**: `sequences_900w.fasta` (original file)
- **VENAS Input**: `sequences_900w_halign4_2019_12-2025_09_20.fasta` (aligned file)
- **Sample Count**: 2,420,470 sequences
- **Processes**: 32 for both tools
- **Reference Sequence**: `NC_045512.2`

## Execution Flow

### TempSnap
- **Command**: `--command all`
- **Complete Pipeline**: rawdata → mcantables → networks → community
- **Input**: Original FASTA file
- **Parallel**: 32 processes

### VENAS
- **Steps**: ePIS finding → Network construction → Community detection
- **Input**: Pre-aligned FASTA file
- **Parallel**: 32 OpenMP threads (`OMP_NUM_THREADS=32`)

## Expected Output Format

The benchmark will generate a CSV report similar to:

```csv
Dataset,Samples,Impl,Processes,Runtime(s),Memory_MB,Peak_Memory_MB,Speedup_vs_VENAS
Sequences_900w,2420470,TEMPNAP,32,1234.56,2048.32,4096.64,1.00
Sequences_900w,2420470,VENAS,32,1851.84,1536.48,3072.96,1.50
```

## Key Features

1. **Separate Input Files**: Each tool uses its optimal input format
2. **32-Process Comparison**: Both tools use maximum parallel processing
3. **Comprehensive Metrics**: Runtime, memory usage, and speedup calculations
4. **Standardized Format**: Compatible with existing benchmark reports

## Running the Benchmark

```bash
cd /home/liujiajun/projects/Hap_networks/module/09-25/Performance_benchmark_results/TempSnap_VS_VENAS
./run_benchmark.sh
```

## Output Files

- `performance_comparison.csv`: Main benchmark results
- `benchmark_comparison.log`: Detailed execution logs
- `tempsnap_results/`: TempSnap output files
- `venas_results/`: VENAS output files

## Technical Notes

- TempSnap uses the complete `all` pipeline for fair comparison
- VENAS uses OpenMP parallel processing controlled by environment variables
- Both tools process the same dataset (2.4M sequences) with 32 parallel workers
- Results include relative speedup calculations for easy comparison
