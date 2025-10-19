#!/usr/bin/env python3
"""
性能比较脚本：TempSnap vs VENAS
比较两种工具在相同数据集上的运行效率
"""

import os
import sys
import time
import subprocess
import pandas as pd
import psutil
from datetime import datetime
import logging
import shutil
import glob

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitor"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.cpu_percentages = []
        
    def start_monitoring(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        logger.info(f"Starting monitoring - Memory usage: {self.start_memory:.2f} MB")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.end_time = time.time()
        self.end_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        logger.info(f"Stopping monitoring - Memory usage: {self.end_memory:.2f} MB")
        
    def get_results(self):
        """Get monitoring results"""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            memory_used = self.end_memory - self.start_memory
            return {
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'memory_used_mb': memory_used,
                'peak_memory_mb': self.end_memory
            }
        return None

class BenchmarkConfig:
    """Benchmark configuration class"""
    
    def __init__(self):
        # === Input file configuration ===
        #self.tempsnap_fasta = "/home/liujiajun/projects/Hap_networks/module/09-25/09-25glob/sequences_900w.fasta"
        self.tempsnap_fasta = "/home/liujiajun/projects/Hap_networks/module/09-25/09-25Mpox/sequences_ncbi.fasta"
        #self.venas_fasta = "/home/liujiajun/projects/Hap_networks/module/09-25/Performance_benchmark_results/TempSnap_VS_VENAS/venas_test/sequences_900w_halign4_2019_12-2025_09_20.fasta"
        self.venas_fasta = "/home/liujiajun/projects/Hap_networks/module/09-25/09-25Mpox/analysed_data/sequences_ncbi_halign4_2022-2025_07_23.fasta"
        self.output_dir = "/home/liujiajun/projects/Hap_networks/module/09-25/Performance_benchmark_results/TempSnap_VS_VENAS/Mpox_results"
        
        # === System configuration ===
        self.num_processes = 16
        #self.sample_count = 2420470  # FASTA sample count
        self.sample_count = 3688
        # === TempSnap parameter configuration ===
        #self.ref_sequence = "NC_045512.2"  # Reference sequence ID
        self.ref_sequence = "NC_003310.1"
        #self.start_date = "2025-09-13"        # Start date (YYYY-MM-DD)
        self.start_date = "2025-06-25"
        #self.end_date = "2025-09-20"          # End date (YYYY-MM-DD)
        self.end_date = "2025-07-23"
        #self.time_interval = 7                # Time interval (days)
        self.time_interval = 28
        #self.max_n_ratio = 0.001              # Maximum N base ratio
        self.max_n_ratio = 0.01
        
        # === Path configuration ===
        self.tempsnap_main_script = "/home/liujiajun/projects/Hap_networks/module/09-25/main.py"
        self.venas_dir = "/home/liujiajun/projects/Hap_networks/VENAS-master"
        
    def get_tempsnap_all_args(self, tempsnap_output_dir):
        """Get TempSnap all command arguments"""
        return [
            'python', self.tempsnap_main_script,
            '--command', 'all',
            '--input_dir', self.tempsnap_fasta,
            '--output_dir', tempsnap_output_dir,
            '--p', str(self.num_processes),
            '--ratio', str(self.max_n_ratio),
            '--ref', self.ref_sequence,
            '--start', self.start_date,
            '--end', self.end_date,
            '--interval', str(self.time_interval)
        ]
    
    def get_venas_args(self, temp_dir, temp_fasta):
        """Get VENAS arguments"""
        return {
            'parsimony_args': [
                'python', '-u', f'{self.venas_dir}/parsimony-informative.py',
                '-i', temp_dir,
                '-m', temp_fasta,
                '-b', 'none',
                '-r', '0',
                '-f', self.ref_sequence
            ],
            'network_args': [
                'python', '-u', f'{self.venas_dir}/haplotype_network.py',
                temp_dir
            ],
            'community_args': [
                'python', f'{self.venas_dir}/main_path_example.py'
            ]
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("Current benchmark configuration:")
        print("=" * 60)
        print(f"TempSnap input file: {self.tempsnap_fasta}")
        print(f"VENAS input file: {self.venas_fasta}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of processes: {self.num_processes}")
        print(f"Sample count: {self.sample_count}")
        print(f"Reference sequence: {self.ref_sequence}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Time interval: {self.time_interval} days")
        print(f"N base ratio: {self.max_n_ratio}")
        print("=" * 60)

# Create global configuration instance
config = BenchmarkConfig()

class BenchmarkRunner:
    """Benchmark runner"""
    
    def __init__(self, output_dir, num_processes=16):
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.results = {}
        self.config = BenchmarkConfig()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Set paths
        self.tempsnap_output = os.path.join(output_dir, 'tempsnap_results')
        self.venas_output = os.path.join(output_dir, 'venas_results')
        self.venas_part1_log = os.path.join(output_dir, 'venas_part1_duration.txt')
        
        # VENAS execution tracking
        self.venas_part1_time = None
        self.venas_part2_start_time = None
        
    def _check_venas_part1_results(self):
        """Check if VENAS Part 1 results exist"""
        part1_files = ['freq_all.txt', 'pi_pos_all.fasta', 'rm_non-ATCG_genomes.ma']
        for file in part1_files:
            if os.path.exists(os.path.join(self.venas_output, file)):
                return True
        return False
    
    def _get_venas_part1_duration(self):
        """Get VENAS Part 1 duration from log file"""
        if os.path.exists(self.venas_part1_log):
            try:
                with open(self.venas_part1_log, 'r') as f:
                    return float(f.read().strip())
            except:
                pass
        return 0
    
    def _save_venas_part1_duration(self, duration):
        """Save VENAS Part 1 duration to log file"""
        with open(self.venas_part1_log, 'w') as f:
            f.write(str(duration))
        
    def run_tempsnap(self):
        """Run TempSnap tool"""
        logger.info("=" * 60)
        logger.info("Starting TempSnap")
        logger.info("=" * 60)
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Create TempSnap output directory
            os.makedirs(self.tempsnap_output, exist_ok=True)
            
            # Run TempSnap with 'all' command to execute complete pipeline
            logger.info("TempSnap: Running complete pipeline with 'all' command")
            cmd = self.config.get_tempsnap_all_args(self.tempsnap_output)
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/liujiajun/projects/Hap_networks/module/09-25')
            
            if result.returncode != 0:
                logger.error(f"TempSnap execution failed: {result.stderr}")
                # Still record performance data even if execution failed
                monitor.stop_monitoring()
                self.results['tempsnap'] = monitor.get_results()
                return False
                
            logger.info("TempSnap execution successful")
            # Record performance data for successful execution
            monitor.stop_monitoring()
            self.results['tempsnap'] = monitor.get_results()
            return True
            
        except Exception as e:
            logger.error(f"TempSnap execution error: {e}")
            # Record performance data even if there was an exception
            monitor.stop_monitoring()
            self.results['tempsnap'] = monitor.get_results()
            return False
            
    def run_venas(self):
        """Run VENAS tool with checkpoint support"""
        logger.info("=" * 60)
        logger.info("Starting VENAS")
        logger.info("=" * 60)
        
        # Check if Part 1 results already exist
        part1_results_exist = self._check_venas_part1_results()
        
        if part1_results_exist:
            logger.info("VENAS Part 1 results found, skipping Part 1 and continuing from Part 2")
            part1_duration = self._get_venas_part1_duration()
            logger.info(f"VENAS Part 1 previously took: {part1_duration:.2f} seconds ({part1_duration/3600:.2f} hours)")
        else:
            logger.info("VENAS Part 1 results not found, starting from Part 1")
            part1_duration = 0
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Set OpenMP thread count for VENAS parallel processing
            os.environ['OMP_NUM_THREADS'] = str(self.num_processes)
            logger.info(f"Set OMP_NUM_THREADS to {self.num_processes} for VENAS parallel processing")
            # Create VENAS output directory
            os.makedirs(self.venas_output, exist_ok=True)
            
            # Copy fasta file to VENAS output directory (only if not exists)
            temp_fasta = os.path.join(self.venas_output, 'sequences.fasta')
            if not os.path.exists(temp_fasta):
                shutil.copy2(self.config.venas_fasta, temp_fasta)
            
            # Find specified reference sequence ID
            ref_id = self.config.ref_sequence  # Use same reference as TempSnap
            ref_found = False
            
            with open(temp_fasta, 'r') as f:
                for line in f:
                    if line.startswith('>') and ref_id in line:
                        ref_found = True
                        break
            
            if not ref_found:
                # If specified reference not found, use first sequence
                logger.warning(f"Reference sequence {ref_id} not found, using first sequence as reference")
                with open(temp_fasta, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('>'):
                        ref_id = first_line[1:]
                    else:
                        ref_id = "reference_sequence"
            
            logger.info(f"Using reference sequence ID: {ref_id}")
            
            # Part 1: ePIS finding (only if results don't exist)
            if not part1_results_exist:
                logger.info("VENAS Part 1: ePIS finding")
                part1_start_time = time.time()
                cmd1 = [
                    'python', '-u', '/home/liujiajun/projects/Hap_networks/VENAS-master/parsimony-informative.py',
                    '-i', self.venas_output,
                    '-m', 'sequences.fasta',
                    '-b', 'none',
                    '-r', '0',
                    '-f', ref_id
                ]
                
                result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd='/home/liujiajun/projects/Hap_networks/VENAS-master')
                if result1.returncode != 0:
                    logger.error(f"VENAS Part 1 execution failed: {result1.stderr}")
                    # Still record performance data even if execution failed
                    monitor.stop_monitoring()
                    self.results['venas'] = monitor.get_results()
                    return False
                    
                part1_duration = time.time() - part1_start_time
                self._save_venas_part1_duration(part1_duration)
                logger.info(f"VENAS Part 1 completed in {part1_duration:.2f} seconds ({part1_duration/3600:.2f} hours)")
            else:
                logger.info("VENAS Part 1: Skipping (results already exist)")
                
            # Part 2: Network construction
            logger.info("VENAS Part 2: Network construction")
            cmd2 = [
                'python', '-u', '/home/liujiajun/projects/Hap_networks/VENAS-master/haplotype_network.py',
                self.venas_output
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd='/home/liujiajun/projects/Hap_networks/VENAS-master')
            if result2.returncode != 0:
                logger.error(f"VENAS Part 2 execution failed: {result2.stderr}")
                # Calculate total time including Part 1
                monitor.stop_monitoring()
                total_duration = monitor.get_results()
                if total_duration:
                    total_duration['duration_seconds'] += part1_duration
                    total_duration['duration_minutes'] = total_duration['duration_seconds'] / 60
                self.results['venas'] = total_duration
                return False
                
            # Part 3: Preprocess network file
            logger.info("VENAS Part 3: Preprocessing network file")
            net_file = os.path.join(self.venas_output, 'net_all.txt')
            csv_file = os.path.join(self.venas_output, 'net.csv')
            
            if os.path.exists(net_file):
                # Convert net_all.txt to CSV format
                with open(net_file, 'r') as infile, open(csv_file, 'w') as outfile:
                    outfile.write('Source,Target\n')
                    for line in infile:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            outfile.write(f"{parts[0]},{parts[1]}\n")
            else:
                logger.error("net_all.txt file not found")
                # Calculate total time including Part 1
                monitor.stop_monitoring()
                total_duration = monitor.get_results()
                if total_duration:
                    total_duration['duration_seconds'] += part1_duration
                    total_duration['duration_minutes'] = total_duration['duration_seconds'] / 60
                self.results['venas'] = total_duration
                return False
                
            # Part 4: Community detection and core node extraction
            logger.info("VENAS Part 4: Community detection and core node extraction")
            cmd4 = [
                'python', '/home/liujiajun/projects/Hap_networks/VENAS-master/main_path_example.py'
            ]
            
            result4 = subprocess.run(cmd4, capture_output=True, text=True, cwd=self.venas_output)
            if result4.returncode != 0:
                logger.error(f"VENAS Part 4 execution failed: {result4.stderr}")
                # Calculate total time including Part 1
                monitor.stop_monitoring()
                total_duration = monitor.get_results()
                if total_duration:
                    total_duration['duration_seconds'] += part1_duration
                    total_duration['duration_minutes'] = total_duration['duration_seconds'] / 60
                self.results['venas'] = total_duration
                return False
                
            # Files are already in the correct output directory
            logger.info("VENAS result files are in the output directory")
                    
            logger.info("VENAS execution successful")
            # Calculate total time including Part 1
            monitor.stop_monitoring()
            total_duration = monitor.get_results()
            if total_duration:
                total_duration['duration_seconds'] += part1_duration
                total_duration['duration_minutes'] = total_duration['duration_seconds'] / 60
                logger.info(f"VENAS total execution time: {total_duration['duration_seconds']:.2f} seconds ({total_duration['duration_seconds']/3600:.2f} hours)")
                logger.info(f"  - Part 1: {part1_duration:.2f} seconds ({part1_duration/3600:.2f} hours)")
                logger.info(f"  - Parts 2-4: {total_duration['duration_seconds'] - part1_duration:.2f} seconds ({(total_duration['duration_seconds'] - part1_duration)/3600:.2f} hours)")
            self.results['venas'] = total_duration
            return True
            
        except Exception as e:
            logger.error(f"VENAS execution error: {e}")
            # Calculate total time including Part 1
            monitor.stop_monitoring()
            total_duration = monitor.get_results()
            if total_duration:
                total_duration['duration_seconds'] += part1_duration
                total_duration['duration_minutes'] = total_duration['duration_seconds'] / 60
            self.results['venas'] = total_duration
            return False
            
    def generate_report(self):
        """Generate performance comparison report"""
        logger.info("=" * 60)
        logger.info("Generating performance comparison report")
        logger.info("=" * 60)
        
        report_data = []
        
        for tool, metrics in self.results.items():
            if metrics:
                report_data.append({
                    'Tool': tool.upper(),
                    'Duration (seconds)': round(metrics['duration_seconds'], 2),
                    'Duration (minutes)': round(metrics['duration_minutes'], 2),
                    'Memory Used (MB)': round(metrics['memory_used_mb'], 2),
                    'Peak Memory (MB)': round(metrics['peak_memory_mb'], 2)
                })
        
        if report_data:
            # Create benchmark-style report similar to SARS_CoV_2_benchmark.csv
            benchmark_data = []
            
            for i, (tool, metrics) in enumerate(self.results.items()):
                if metrics:
                    benchmark_data.append({
                        'Dataset': 'Sequences_900w',
                        'Samples': self.config.sample_count,
                        'Impl': tool.upper(),
                        'Processes': self.num_processes,
                        'Runtime(s)': round(metrics['duration_seconds'], 2),
                        'Memory_MB': round(metrics['memory_used_mb'], 2),
                        'Peak_Memory_MB': round(metrics['peak_memory_mb'], 2)
                    })
            
            # Calculate speedup relative to slower tool (only if both tools have data)
            if len(benchmark_data) == 2:
                tempsnap_time = benchmark_data[0]['Runtime(s)']
                venas_time = benchmark_data[1]['Runtime(s)']
                
                if tempsnap_time > venas_time:
                    # TempSnap is slower, VENAS is faster
                    benchmark_data[0]['Speedup_vs_VENAS'] = round(tempsnap_time / venas_time, 2)
                    benchmark_data[1]['Speedup_vs_VENAS'] = 1.0
                else:
                    # VENAS is slower, TempSnap is faster
                    benchmark_data[1]['Speedup_vs_VENAS'] = round(venas_time / tempsnap_time, 2)
                    benchmark_data[0]['Speedup_vs_VENAS'] = 1.0
            elif len(benchmark_data) == 1:
                # Only one tool has data, set speedup to N/A
                benchmark_data[0]['Speedup_vs_VENAS'] = 'N/A'
            
            df = pd.DataFrame(benchmark_data)
            
            # Save CSV report
            report_file = os.path.join(self.output_dir, 'performance_comparison.csv')
            df.to_csv(report_file, index=False)
            logger.info(f"Performance report saved to: {report_file}")
            
            # Print report
            print("\n" + "=" * 100)
            print("Performance Comparison Report (32 Processes)")
            print("=" * 100)
            print(df.to_string(index=False))
            
            # Print summary
            if len(benchmark_data) == 2:
                faster_tool = benchmark_data[0]['Impl'] if benchmark_data[0]['Runtime(s)'] < benchmark_data[1]['Runtime(s)'] else benchmark_data[1]['Impl']
                slower_tool = benchmark_data[1]['Impl'] if faster_tool == benchmark_data[0]['Impl'] else benchmark_data[0]['Impl']
                speedup = max(benchmark_data[0]['Speedup_vs_VENAS'], benchmark_data[1]['Speedup_vs_VENAS'])
                print(f"\n{faster_tool} is faster than {slower_tool} with {speedup:.2f}x speedup")
            elif len(benchmark_data) == 1:
                tool_name = benchmark_data[0]['Impl']
                runtime = benchmark_data[0]['Runtime(s)']
                print(f"\nOnly {tool_name} data available - Runtime: {runtime} seconds")
                
            return df
        else:
            logger.error("No performance data available")
            return None

def main():
    """Main function"""
    # Use parameters from config file
    output_dir = config.output_dir
    num_processes = config.num_processes
    
    # Print configuration information
    config.print_config()
    
    logger.info(f"Starting performance comparison test")
    logger.info(f"TempSnap input file: {config.tempsnap_fasta}")
    logger.info(f"VENAS input file: {config.venas_fasta}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of processes: {num_processes}")
    
    # Check if input files exist
    if not os.path.exists(config.tempsnap_fasta):
        logger.error(f"TempSnap input file does not exist: {config.tempsnap_fasta}")
        return
        
    if not os.path.exists(config.venas_fasta):
        logger.error(f"VENAS input file does not exist: {config.venas_fasta}")
        return
        
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir, num_processes)
    
    # Run TempSnap first and record performance data
    logger.info("=" * 80)
    logger.info("EXECUTING TEMPSNAP")
    logger.info("=" * 80)
    tempsnap_success = runner.run_tempsnap()
    
    # Always record TempSnap performance data regardless of success/failure
    if 'tempsnap' in runner.results:
        logger.info(f"TempSnap performance recorded: {runner.results['tempsnap']}")
    else:
        logger.warning("TempSnap performance data not recorded")
    
    # Run VENAS second and record performance data
    logger.info("=" * 80)
    logger.info("EXECUTING VENAS")
    logger.info("=" * 80)
    venas_success = runner.run_venas()
    
    # Always record VENAS performance data regardless of success/failure
    if 'venas' in runner.results:
        logger.info(f"VENAS performance recorded: {runner.results['venas']}")
    else:
        logger.warning("VENAS performance data not recorded")
    
    # Generate report if we have any performance data
    if runner.results:
        logger.info("=" * 80)
        logger.info("GENERATING PERFORMANCE REPORT")
        logger.info("=" * 80)
        report = runner.generate_report()
        if report is not None:
            logger.info("Performance comparison completed")
            if tempsnap_success and venas_success:
                logger.info("Both tools executed successfully")
            elif tempsnap_success:
                logger.warning("Only TempSnap executed successfully")
            elif venas_success:
                logger.warning("Only VENAS executed successfully")
            else:
                logger.warning("Both tools failed, but performance data was recorded")
        else:
            logger.error("Report generation failed")
    else:
        logger.error("No performance data available for report generation")
        
    # No temporary files to clean up - all files are in result directories

if __name__ == "__main__":
    main()
