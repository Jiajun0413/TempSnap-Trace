# v 2025-04-15
import collections
import datetime as dt
import io
import logging
import os
import re
import gc
import warnings
import multiprocessing
from contextlib import closing
from typing import List, Tuple, Optional, Any, Dict
import blosc2
import h5py
import igraph as ig
import joblib
import numpy as np
import pandas as pd
import psutil
import McAN 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

# --- Constants ---
DEFAULT_PROCESS_COUNT = 4
HDF5_CACHE_SIZE_MB = 128
HDF5_MAX_CACHE_FRACTION = 0.25
DEFAULT_COMPRESSION_LEVEL = 3
DEFAULT_WDKS_VALUE = 1e-6
INFOMAP_TRIALS = 20
ARROW_COMPRESSION = 'lz4'
BLOSC_CODEC = blosc2.Codec.LZ4
JOB_PROTOCOL = 5

# --- Custom Log Filter ---
class ConsoleFilter(logging.Filter):
    """Prevents log records with 'file_only=True' from reaching the console."""
    def filter(self, record):
        return not getattr(record, 'file_only', False)

class LogManager:
    """Manages logging configuration."""

    @staticmethod
    def configure_logging(output_dir: Optional[str] = None,
                          process_type: str = "main",
                          append_mode: bool = False) -> int:
        """Configures the logging system."""
        root_logger = logging.getLogger()
        
        # 避免重复配置
        if hasattr(root_logger, '_configured'):
            return os.getpid()
            
        root_logger.setLevel(logging.INFO)
        
        # 移除所有现有的处理器
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        
        # 添加控制台处理器（带有过滤器）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        console_handler.addFilter(ConsoleFilter())  # 添加过滤器
        root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "log.txt")
            mode = 'a' if append_mode or process_type == "subprocess" else 'w'
            file_handler = logging.FileHandler(log_file, mode=mode)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            root_logger.addHandler(file_handler)
        
        # 标记为已配置
        setattr(root_logger, '_configured', True)
        return os.getpid()

class IOManager:
    """Handles optimized data I/O using HDF5 and Blosc2."""


    @staticmethod
    def _serialize_data(data: Any) -> Tuple[bytes, str]:
        """Serialize data using the most appropriate method."""
        # ... (no changes needed in this method) ...
        if isinstance(data, pd.DataFrame):
            try:
                with io.BytesIO() as buffer:
                    data.to_feather(buffer, compression=ARROW_COMPRESSION)
                    return buffer.getvalue(), 'feather'
            except Exception:
                # Fallback for DataFrame: convert object columns and retry feather
                try:
                    df_copy = data.copy()
                    for col in df_copy.select_dtypes(include=['object']).columns:
                        df_copy[col] = df_copy[col].astype(str)
                    with io.BytesIO() as buffer:
                        df_copy.to_feather(buffer, compression=ARROW_COMPRESSION)
                        return buffer.getvalue(), 'feather_converted'
                except Exception as e:
                    logging.warning(f"Feather serialization failed even after conversion: {e}. Falling back to joblib.")
                    # Further fallback to joblib
                    with io.BytesIO() as buffer:
                        joblib.dump(data, buffer, compress=False, protocol=JOB_PROTOCOL)
                        return buffer.getvalue(), 'joblib'
        elif isinstance(data, ig.Graph):
            with io.BytesIO() as buffer:
                joblib.dump(data, buffer, compress=False, protocol=JOB_PROTOCOL)
                return buffer.getvalue(), 'joblib'
        else: # Generic data
            with io.BytesIO() as buffer:
                joblib.dump(data, buffer, compress=False, protocol=JOB_PROTOCOL)
                return buffer.getvalue(), 'joblib'

    @staticmethod
    def _deserialize_data(binary_data: bytes, data_format: str, content_type: str) -> Any:
        """Deserialize data based on format."""
        # ... (no changes needed in this method) ...
        try:
            with io.BytesIO(binary_data) as bio:
                if data_format.startswith('feather'):
                    return pd.read_feather(bio)
                elif data_format == 'joblib':
                    return joblib.load(bio)
                else: # Fallback attempt for unknown format
                    if content_type == 'dataframe_list':
                         return pd.read_feather(bio) # Try feather first
                    else:
                         return joblib.load(bio) # Try joblib
        except Exception as e:
            logging.error(f"Deserialization failed for format '{data_format}', content '{content_type}': {e}", exc_info=False)
            return None

    @staticmethod
    def save_to_hdf5(data: Any, filepath: str, num_threads: int = DEFAULT_PROCESS_COUNT, compression_level: int = DEFAULT_COMPRESSION_LEVEL) -> str:
        """Saves data to an HDF5 file with optimized compression."""
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if not filepath.endswith('.h5'):
            filepath = os.path.splitext(filepath)[0] + '.h5'

        # Set blosc threads based on the provided argument
        blosc_threads = max(1, num_threads) # Ensure at least 1 thread
        blosc2.set_nthreads(blosc_threads)
        cache_size = min(int(psutil.virtual_memory().available * HDF5_MAX_CACHE_FRACTION),
                         HDF5_CACHE_SIZE_MB * 1024 * 1024)

        logging.info(f"Saving data to {os.path.basename(filepath)} (CompLevel:{compression_level}, Threads:{blosc_threads})")

        try:
            with h5py.File(filepath, 'w', libver='latest', rdcc_nbytes=cache_size) as f:
                f.attrs['creation_timestamp'] = dt.datetime.now(dt.timezone.utc).isoformat()

                if isinstance(data, list):
                    f.attrs['content_type'] = 'list'
                    f.attrs['list_length'] = len(data)
                    list_item_type = None
                    if data:
                        for item in data:
                            if item is not None:
                                list_item_type = type(item).__name__
                                break
                    f.attrs['list_item_type'] = list_item_type if list_item_type else 'Unknown'

                    for i, item in enumerate(data):
                        item_key = f'item_{i}'
                        if item is None or (hasattr(item, 'empty') and item.empty) or \
                           (isinstance(item, ig.Graph) and item.vcount() == 0):
                            f.create_dataset(item_key, data=h5py.Empty("f"))
                            f[item_key].attrs['format'] = 'empty'
                            continue

                        try:
                            binary_data, data_format = IOManager._serialize_data(item)
                            # Compression uses the globally set thread count
                            compressed_data = blosc2.compress(binary_data, typesize=1, clevel=compression_level, codec=BLOSC_CODEC)
                            dset = f.create_dataset(item_key, data=np.frombuffer(compressed_data, dtype=np.uint8), chunks=True)
                            dset.attrs['format'] = data_format
                            dset.attrs['original_size_bytes'] = len(binary_data)
                            del binary_data, compressed_data
                        except Exception as e:
                            logging.error(f"Failed to save item {i}: {e}", exc_info=False)
                            if item_key not in f:
                                f.create_dataset(item_key, data=h5py.Empty("f"))
                                f[item_key].attrs['format'] = 'error'

                else: # Handle single generic object
                    f.attrs['content_type'] = 'generic'
                    try:
                        binary_data, data_format = IOManager._serialize_data(data)
                        # Compression uses the globally set thread count
                        compressed_data = blosc2.compress(binary_data, typesize=1, clevel=compression_level, codec=BLOSC_CODEC)
                        dset = f.create_dataset('data_payload', data=np.frombuffer(compressed_data, dtype=np.uint8), chunks=True)
                        dset.attrs['format'] = data_format
                        dset.attrs['original_size_bytes'] = len(binary_data)
                        del binary_data, compressed_data
                    except Exception as e:
                        logging.error(f"Failed to save generic data: {e}", exc_info=True)
                        raise

        except Exception as e:
            logging.error(f"Error saving HDF5 file {filepath}: {e}", exc_info=True)
            raise

        #logging.info(f"Successfully saved {os.path.basename(filepath)}")
        return filepath

    @staticmethod
    def load_from_hdf5(filepath: str, parallelism_level: int = DEFAULT_PROCESS_COUNT) -> Any:
        """Loads data from an HDF5 file, using parallel processing for lists. (Simplified Version)"""
        if not filepath.endswith('.h5'):
            filepath = os.path.splitext(filepath)[0] + '.h5'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        start_time = dt.datetime.now()
        filename = os.path.basename(filepath)
        # Reduced cache size for potentially simpler use cases, adjust if needed
        io_config = {'libver': 'latest', 'rdcc_nbytes': 64 * 1024 * 1024, 'swmr': True}
        blosc_threads = max(1, parallelism_level)
        final_result = None

        try:
            with h5py.File(filepath, 'r', **io_config) as f:
                content_type = f.attrs.get('content_type')
                #logging.info(f"Loading HDF5 file: {filename} (Type: {content_type})")

                if content_type == 'generic' and 'data_payload' in f:
                    dset = f['data_payload']
                    data_format = dset.attrs.get('format', 'joblib')
                    compressed_bytes = np.frombuffer(dset[:], dtype='uint8').tobytes()
                    blosc2.set_nthreads(blosc_threads)
                    decompressed_data = blosc2.decompress(compressed_bytes)
                    result = IOManager._deserialize_data(decompressed_data, data_format, 'generic')
                    del compressed_bytes, decompressed_data
                    final_result = result

                elif content_type == 'list':
                    list_length = f.attrs.get('list_length', 0)
                    list_item_type_attr = f.attrs.get('list_item_type', 'Unknown') # Still useful to know

                    if list_length == 0:
                        logging.warning(f"File '{filename}' has list_length=0. Returning empty list.")
                        final_result = []
                    else:
                        num_workers = max(1, parallelism_level)
                        #logging.info(f"Loading list ({list_item_type_attr}, {list_length} items) from {filename} using {num_workers} processes.")

                        chunk_sizes = ParallelProcessor.get_chunk_sizes(list_length, num_workers)
                        task_args = []
                        start_idx = 0
                        for size in chunk_sizes:
                            end_idx = start_idx + size
                            task_args.append({
                                'range': (start_idx, end_idx),
                                'filepath': filepath,
                                'metadata': {
                                    'content_type': content_type,
                                    'list_item_type': list_item_type_attr,
                                    'io_config': io_config,
                                    'num_threads': blosc_threads
                                }
                            })
                            start_idx = end_idx

                        with closing(ParallelProcessor.get_process_pool(processes=num_workers, maxtasksperchild=1)) as pool:
                            all_results_map = {}
                            # Use imap_unordered for potentially better performance with uneven task times
                            for chunk_result_map in pool.imap_unordered(IOManager._process_chunk, task_args):
                                all_results_map.update(chunk_result_map)

                        loaded_list = [all_results_map.get(i) for i in range(list_length)]
                        final_result = loaded_list

                else:
                    raise ValueError(f"Unsupported HDF5 content type: {content_type}")

            # Log completion time
            logging.info(f"Finished loading {filename} in {(dt.datetime.now() - start_time).total_seconds():.2f}s")
            return final_result

        except Exception as e:
            logging.error(f"Error loading HDF5 file {filepath}: {e}", exc_info=True)
            raise

    @staticmethod
    def _process_chunk(args: Dict[str, Any]) -> Dict[int, Any]:
        """Processes a chunk of items from an HDF5 file list (run in a subprocess)."""
        # --- This function remains largely the same as it's crucial for parallel loading ---
        idx_range, filepath, metadata = args['range'], args['filepath'], args['metadata']
        start_idx, end_idx = idx_range
        content_type = metadata['content_type']
        list_item_type = metadata.get('list_item_type', 'Unknown')
        io_config = metadata.get('io_config', {})
        num_threads = metadata.get('num_threads', 1)
        results = {}
        pid = os.getpid()

        # Optional: Set CPU affinity (can be kept or removed for simplicity)
        try:
            if hasattr(os, 'sched_setaffinity'):
                cpu_count = os.cpu_count() or 1
                affinity = {pid % cpu_count}
                os.sched_setaffinity(0, affinity)
        except Exception:
            pass

        blosc2.set_nthreads(max(1, num_threads))

        try:
            with h5py.File(filepath, 'r', **io_config) as f:
                for i in range(start_idx, end_idx):
                    item_key = f'item_{i}'
                    if item_key not in f:
                        results[i] = None
                        continue
                    try:
                        dset = f[item_key]
                        data_format = dset.attrs.get('format', 'unknown')

                        if data_format == 'empty' or dset.shape is None or dset.shape == ():
                            results[i] = None
                            continue
                        if data_format == 'error':
                             # Keep warning for skipped error items
                             logging.warning(f"PID {pid}: Skipping item {i} due to previous save error.")
                             results[i] = None
                             continue

                        compressed_bytes = np.frombuffer(dset[:], dtype='uint8').tobytes()
                        decompressed_data = blosc2.decompress(compressed_bytes)
                        results[i] = IOManager._deserialize_data(decompressed_data, data_format, content_type)
                        del compressed_bytes, decompressed_data
                    except Exception as item_e:
                        # Keep logging for individual item failures
                        logging.error(f"PID {pid}: Failed to process item {i}: {item_e}", exc_info=False)
                        results[i] = None
        except Exception as file_e:
            logging.error(f"PID {pid}: Failed to open/read {os.path.basename(filepath)} in subprocess: {file_e}", exc_info=True)
            results = {i: None for i in range(start_idx, end_idx)}

        gc.collect()
        return results

class NetworkUtils:
    """Provides graph analysis utility functions."""

    @staticmethod
    def compute_wdks(g: ig.Graph):
        """Calculates Weighted Degree k-shell (WDKS) for nodes according to the paper."""
        if g.vcount() == 0:
            return
        
        try:
            # Calculate the degree of each node (total degree: in-degree + out-degree)
            degrees = np.array(g.degree(mode='all'))
            
            # Calculate the k-shell value (based on the total degree)
            kshell_indices = np.array(g.shell_index(mode='all'))
            kshell_indices = np.maximum(1, kshell_indices) * (degrees > 0)  # Ensure k-shell ≥1 and degree > 0
            
            # Calculate the intrinsic influence of each node: k_i * ks_i
            node_prop_product = degrees * kshell_indices
            
            # If there are no edges, directly return the intrinsic part
            if g.ecount() == 0:
                g.vs['wdks'] = node_prop_product.tolist()
                return
            
            # Process edges (keep the direction for directed graphs, remove duplicates for undirected graphs)
            edges = []
            if g.is_directed():
                # Directed graph: keep all original edges
                edges = [(e.source, e.target) for e in g.es]
            else:
                # Undirected graph: standardize edges and remove duplicates (e.g., (u,v) and (v,u) are considered the same edge)
                edges = [tuple(sorted((e.source, e.target))) for e in g.es]
                edges = list(set(edges))  # Remove duplicates
            
            # Extract source and target node indices
            if edges:
                source_indices, target_indices = zip(*edges)
            else:
                source_indices, target_indices = [], []
            
            # Calculate the edge weight w_ij = (k_u * ks_u) * (k_v * ks_v)
            edge_weights = node_prop_product[list(source_indices)] * node_prop_product[list(target_indices)]
            avg_edge_weight = np.mean(edge_weights) if edge_weights.size > 0 else 0
            
            # Calculate the extrinsic part contribution
            extrinsic_part = np.zeros(g.vcount(), dtype=float)
            if avg_edge_weight != 0:
                norm_weights = edge_weights / avg_edge_weight
                for (u, v), w in zip(edges, norm_weights):
                    # For each edge, add the contribution to both nodes (regardless of directed/undirected)
                    extrinsic_part[u] += w * node_prop_product[v]
                    extrinsic_part[v] += w * node_prop_product[u]
            
            # Final WDKS = intrinsic + extrinsic
            wdks_values = node_prop_product + extrinsic_part
            
            # Store the results in the graph properties
            g.vs['wdks'] = wdks_values.tolist()
        
        except Exception as e:
            logging.error(f"Error calculating WDKS: {e}", exc_info=False)
            g.vs['wdks'] = [0.0] * g.vcount()

    @staticmethod
    def calculate_jaccard_similarity(str1: Optional[str], str2: Optional[str], delimiter: str = ';') -> float:
        """Calculates Jaccard similarity between two delimited strings."""
        try:
            def extract_items(item_str: Optional[str]) -> set:
                return set(item.strip() for item in item_str.split(delimiter) if item and item.strip()) if isinstance(item_str, str) else set()

            set1, set2 = extract_items(str1), extract_items(str2)
            if not set1 and not set2: return 1.0
            union_size = len(set1.union(set2))
            return len(set1.intersection(set2)) / union_size if union_size > 0 else 0.0
        except Exception:
            # logging.error(f"Jaccard calculation error: {e}", exc_info=False) # Keep logs cleaner
            return 0.5 # Neutral value on error

class ParallelProcessor:
    """Manages multiprocessing pools."""

    @staticmethod
    def get_process_pool(processes: Optional[int] = None,
                         maxtasksperchild: Optional[int] = 1,
                         context: str = 'spawn'):
        """Creates a multiprocessing Pool."""
        num_processes = max(1, processes if processes is not None else DEFAULT_PROCESS_COUNT)
        gc.collect()
        try:
            # logging.debug(f"Creating pool: {num_processes} workers, context='{context}', maxtasks={maxtasksperchild}")
            return multiprocessing.get_context(context).Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)
        except Exception as e:
            logging.error(f"Pool creation failed (context '{context}'): {e}. Trying fallback.")
            fallback_context = 'fork' if context != 'fork' else 'spawn'
            try:
                return multiprocessing.get_context(fallback_context).Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)
            except Exception as e2:
                logging.critical(f"Fallback pool creation failed: {e2}", exc_info=True)
                raise RuntimeError("Failed to create multiprocessing pool.") from e2

    @staticmethod
    def get_chunk_sizes(total_items: int, num_chunks: int) -> List[int]:
        """Divides items into roughly equal chunks."""
        if num_chunks <= 0 or total_items <= 0: return []
        num_chunks = min(total_items, num_chunks)
        base_size, remainder = divmod(total_items, num_chunks)
        return [base_size + 1] * remainder + [base_size] * (num_chunks - remainder)

class TempSnap:
    """Analyzes temporal networks, performs community detection, and extracts backbones."""

    def __init__(self,
                 output_path: str,
                 samples_df: Optional[pd.DataFrame] = None,
                 samples_path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 time_interval: Optional[Any] = None,
                 num_processes: Optional[int] = None,
                 optional_attrs: Optional[List[str]] = None):
        """Initializes the TempSnap analyzer."""
        if not output_path: raise ValueError("Output path must be provided.")
        self.output_path = output_path
        self.samples_df = samples_df
        self.samples_path = samples_path
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.num_processes = max(1, num_processes if num_processes is not None else DEFAULT_PROCESS_COUNT)
        self.optional_attrs = optional_attrs if optional_attrs else []
        self.raw_networks_data: List[Optional[pd.DataFrame]] = []
        self.temporal_graphs: List[Optional[ig.Graph]] = []

        os.makedirs(self.output_path, exist_ok=True)
        LogManager.configure_logging(self.output_path, process_type="main") # Configure main logging once

    @staticmethod
    def _configure_subprocess_logging(output_path: str) -> int:
        """Configures logging for a subprocess (appends)."""
        return LogManager.configure_logging(output_path, process_type="subprocess", append_mode=True)

    # --- Static Wrappers for Parallel Processing ---
    @staticmethod
    def _build_graph_task(args: Tuple[pd.DataFrame, int, str, List[str]]) -> Tuple[int, Optional[ig.Graph]]:
        """Serializable wrapper for build_single_graph."""
        df, index, output_path, optional_attrs = args
        pid = TempSnap._configure_subprocess_logging(output_path)
        try:
            graph = TempSnap.build_single_graph(df, index, optional_attrs)
            return (index, graph)
        except Exception as e:
            logging.error(f"PID {pid}: Error building graph {index}: {e}", exc_info=False)
            return (index, None)
        finally:
            pass

    @staticmethod
    def _detect_partition_task(args: Tuple[ig.Graph, int, Optional[pd.DataFrame], str, List[str]]) -> Tuple[int, Tuple[List[List[str]], Tuple[float, float], Optional[ig.Graph], pd.DataFrame]]:
        """Serializable wrapper for detect_partition_and_build_backbone."""
        graph, index, df, output_path, optional_attrs = args
        pid = TempSnap._configure_subprocess_logging(output_path)
        try:
            result = TempSnap.detect_partition_and_build_backbone(graph, index, df, optional_attrs)
            return (index, result)
        except Exception as e:
            logging.error(f"PID {pid}: Error detecting partition {index}: {e}", exc_info=False)
            fallback_communities = [[v['name'] for v in graph.vs]] if graph and graph.vcount() > 0 else []
            return (index, (fallback_communities, (0.0, float('nan')), None, pd.DataFrame()))
        finally:
            pass

    # --- Core Workflow Methods ---
    def run_mcan_simulation(self, n_rows: Optional[int] = None) -> str:
        """Runs McAN simulation to generate raw network data."""
        if not self.samples_path and self.samples_df is None:
             raise ValueError("Samples path or DataFrame required for McAN.")
        if not self.start_date or not self.end_date or not self.time_interval:
             raise ValueError("Date range and time interval required for McAN.")

        logging.info(f"Starting McAN simulation: {self.start_date} to {self.end_date}")
        start_sim_time = dt.datetime.now()

        try:
            if self.samples_path and self.samples_df is None:
                logging.info(f"Reading samples from: {self.samples_path}" + (f" (limit {n_rows} rows)" if n_rows else ""))
                self.samples_df = pd.read_csv(self.samples_path, nrows=n_rows, sep='\t', low_memory=False)

            if self.samples_df is None or self.samples_df.empty:
                 raise ValueError("Sample data is empty or could not be loaded.")

            simulator = McAN.Calcdate(samples=self.samples_df, time_interval=self.time_interval,
                                      start_date=self.start_date, end_date=self.end_date)
            mcan_results_list = []
            step_count = 0
            current_sim_date = simulator.start_date

            while current_sim_date <= simulator.end_date:
                step_start_time = dt.datetime.now()
                simulator.current_date = current_sim_date
                simulator.update_data_available()

                if simulator.samples_available is None or simulator.samples_available.empty:
                    logging.warning(f"No samples for {current_sim_date}. Skipping.")
                    mcan_results_list.append(None)
                else:
                    mcan_instance = McAN.McAN(output_path=self.output_path, samples=simulator.samples_available,
                                             current_date=current_sim_date, time_interval=simulator.time_interval)
                    mcan_instance.mcan(num_of_processes=self.num_processes)
                    mcan_results_list.append(getattr(mcan_instance, 'df_haps', None))

                logging.info(f"McAN step {current_sim_date} finished in {(dt.datetime.now() - step_start_time).total_seconds():.2f}s",extra={'file_only': True}) # Reduce verbosity
                simulator.next_date()
                current_sim_date = simulator.current_date
                step_count += 1

            logging.info(f"McAN simulation completed {step_count} steps in {(dt.datetime.now() - start_sim_time).total_seconds():.2f}s.")
            self.raw_networks_data = mcan_results_list
            output_filepath = os.path.join(self.output_path, f"McAN_raw_results_{self.start_date}_to_{self.end_date}.h5")
            IOManager.save_to_hdf5(self.raw_networks_data, output_filepath, num_threads=self.num_processes)
            return output_filepath

        except (FileNotFoundError, ValueError, ImportError) as e:
             logging.error(f"McAN simulation failed: {e}", exc_info=True)
             raise
        except Exception as e:
            logging.error(f"Unexpected error during McAN simulation: {e}", exc_info=True)
            raise

    @staticmethod
    def build_single_graph(df: Optional[pd.DataFrame], snapshot_index: int, optional_attrs: List[str]) -> Optional[ig.Graph]:
        """Builds a single directed graph for a time snapshot."""
        if df is None or df.empty:
            return None
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
            if df.empty: return None

        try:
            df.index = df.index.astype(str)
            # Simplified Ancestor processing
            if 'Ancestor' in df.columns:
                df['Ancestor'] = df['Ancestor'].apply(
                    lambda x: [str(a) for a in x if pd.notna(a) and str(a)] if isinstance(x, (list, np.ndarray))
                    else [str(x)] if pd.notna(x) and str(x) else []
                )
            else:
                 df['Ancestor'] = [[] for _ in range(len(df))]

            g = ig.Graph(directed=True)
            vertex_names = list(df.index)
            g.add_vertices(vertex_names)
            if g.vcount() == 0: return None

            # Batch assign attributes (more concise)
            base_attrs = ['Date', 'Location', 'ID', 'Ancestor_ID', 'Lineage', 'Clade', 'Ancestor', 'size']
            for attr in base_attrs + optional_attrs:
                if attr in df.columns:
                    try:
                        g.vs[attr] = df[attr].reindex(vertex_names).tolist()
                    except Exception as e:
                         logging.warning(f"Snapshot {snapshot_index}: Error adding attr '{attr}': {e}")
                         g.vs[attr] = [None] * g.vcount()

            edges, jaccard_weights = [], []
            mutation_col = 'mutation_str'
            use_mutation_str = mutation_col in df.columns

            for hap_id, ancestors in df['Ancestor'].items():
                for parent_id in ancestors:
                    if parent_id in df.index:
                        if use_mutation_str:
                            parent_feat = df.loc[parent_id, mutation_col]
                            child_feat = df.loc[hap_id, mutation_col]
                        else:
                            parent_feat, child_feat = parent_id, hap_id

                        weight = NetworkUtils.calculate_jaccard_similarity(parent_feat, child_feat)
                        edges.append((parent_id, hap_id))
                        jaccard_weights.append(weight)

            initial_vcount = g.vcount()
            if edges: g.add_edges(edges); g.es['evo_weight'] = jaccard_weights
            isolated_indices = g.vs.select(_degree_eq=0).indices
            num_removed = 0 # Track removed nodes
            if isolated_indices:
                num_removed = len(isolated_indices)
                g.delete_vertices(isolated_indices)

            final_vcount, final_ecount = g.vcount(), g.ecount()
            if final_ecount > 0:
                NetworkUtils.compute_wdks(g)
                if 'wdks' in g.vertex_attributes():
                    wdks_raw = np.array(g.vs['wdks'], dtype=float)
                    valid_mask = np.isfinite(wdks_raw)
                    if np.any(valid_mask):
                        min_val, max_val = np.min(wdks_raw[valid_mask]), np.max(wdks_raw[valid_mask])
                        wdks_range = max_val - min_val
                        norm_wdks = (wdks_raw - min_val) / wdks_range if wdks_range > 1e-9 else 0.5
                        final_wdks = np.maximum(norm_wdks, DEFAULT_WDKS_VALUE)
                        final_wdks[~valid_mask] = DEFAULT_WDKS_VALUE
                        g.vs['wdks'] = final_wdks.tolist()
                    else: g.vs['wdks'] = [DEFAULT_WDKS_VALUE] * final_vcount
                else: g.vs['wdks'] = [DEFAULT_WDKS_VALUE] * final_vcount
            elif final_vcount > 0:
                g.vs['wdks'] = [DEFAULT_WDKS_VALUE] * final_vcount

            # Log only to file: include number of removed nodes
            logging.info(f"Snapshot {snapshot_index}: Built graph {final_vcount}v/{final_ecount}e (removed {num_removed} isolated nodes)", extra={'file_only': True})
            return g

        except Exception as e:
            logging.error(f"Snapshot {snapshot_index}: Error building graph: {e}", exc_info=True)
            return None
        
    def _load_or_get_data(self, file_path: Optional[str], current_data: List, data_type: str, default_pattern: str) -> Tuple[List, Optional[str]]:
        """Helper to load data from HDF5 or use existing data."""
        if file_path:
            #logging.info(f"Loading {data_type} from: {os.path.basename(file_path)}")
            # Pass self.num_processes to load_from_hdf5 as parallelism_level
            loaded_data = IOManager.load_from_hdf5(file_path, parallelism_level=self.num_processes)
            # ... (rest of the method) ...
            match = re.search(default_pattern.replace('{start}', r'(\d{4}-\d{2}-\d{2})').replace('{end}', r'(\d{4}-\d{2}-\d{2})'), os.path.basename(file_path))
            if match:
                self.start_date = self.start_date or match.group(1)
                self.end_date = self.end_date or match.group(2)
            return loaded_data, file_path
        elif current_data:
            logging.info(f"Using pre-loaded {data_type}.")
            return current_data, None
        elif self.start_date and self.end_date:
            default_path = os.path.join(self.output_path, default_pattern.format(start=self.start_date, end=self.end_date))
            if os.path.exists(default_path):
                logging.warning(f"No {data_type} provided, loading default: {os.path.basename(default_path)}")
                return self._load_or_get_data(default_path, [], data_type, default_pattern) # Recursive call
            else:
                raise ValueError(f"{data_type.capitalize()} are required but none provided or found.")
        else:
            raise ValueError(f"{data_type.capitalize()} are required, but none provided and date range unknown.")

    def build_temporal_graphs(self, raw_results_path: Optional[str] = None) -> str:
        """Builds graphs for each time snapshot in parallel."""
        logging.info("Starting temporal graph construction...")
        start_build_time = dt.datetime.now()

        source_data, _ = self._load_or_get_data(
            raw_results_path, self.raw_networks_data, "raw McAN results",
            "McAN_raw_results_{start}_to_{end}.h5"
        )

        if not isinstance(source_data, list) or not source_data:
            logging.warning("Source data empty. No graphs built.")
            self.temporal_graphs = []
            output_filepath = os.path.join(self.output_path, f"Temporal_graphs_{self.start_date}_to_{self.end_date}.h5")
            IOManager.save_to_hdf5([], output_filepath)
            return output_filepath

        tasks = [(df, i, self.output_path, self.optional_attrs) for i, df in enumerate(source_data)]
        num_tasks = len(tasks)
        logging.info(f"Building {num_tasks} graphs using {self.num_processes} processes.")

        built_graphs_map = {}
        try:
            with closing(ParallelProcessor.get_process_pool(processes=self.num_processes, maxtasksperchild=1)) as pool:
                processed_count = 0
                last_milestone = 0
                for index, graph in pool.imap_unordered(TempSnap._build_graph_task, tasks,
                                                        chunksize=max(1, num_tasks // (self.num_processes * 2))):
                    built_graphs_map[index] = graph
                    processed_count += 1
                    current_percent = (processed_count * 100) // num_tasks
                    current_milestone = (current_percent // 20) * 20
                    if current_milestone > last_milestone or processed_count == num_tasks:
                        logging.info(f"Graph building progress: {current_milestone}% ({processed_count}/{num_tasks})")
                        last_milestone = current_milestone
        except Exception as e:
            logging.error(f"Parallel graph building failed: {e}", exc_info=True)
            raise

        self.temporal_graphs = [built_graphs_map.get(i) for i in range(num_tasks)]
        valid_graph_count = sum(1 for g in self.temporal_graphs if g is not None)
        logging.info(f"Built {valid_graph_count}/{num_tasks} graphs in {(dt.datetime.now() - start_build_time).total_seconds():.2f}s.")

        output_filepath = os.path.join(self.output_path, f"Temporal_graphs_{self.start_date}_to_{self.end_date}.h5")
        IOManager.save_to_hdf5(self.temporal_graphs, output_filepath, num_threads=self.num_processes)
        return output_filepath

    @staticmethod
    def detect_partition_and_build_backbone(
            g: Optional[ig.Graph], snapshot_index: int, df: Optional[pd.DataFrame] = None, optional_attrs: List[str] = []
            ) -> Tuple[List[List[str]], Tuple[float, float], Optional[ig.Graph], pd.DataFrame]:
        """Performs community detection (Infomap) and extracts backbone network."""
        if g is None or g.vcount() < 1:
            return [], (0.0, float('nan')), None, pd.DataFrame()
        if g.ecount() == 0:
            communities = [[v['name']] for v in g.vs]
            node_data = [{'node_id': v['name'], 'is_key_node': True} | {attr: v[attr] for attr in v.attributes() if attr != 'name'} for v in g.vs]
            backbone_df = pd.DataFrame(node_data).set_index('node_id') if node_data else pd.DataFrame()
            return communities, (0.0, float('nan')), None, backbone_df

        try:
            weights_arg = {"edge_weights": "evo_weight"} if "evo_weight" in g.edge_attributes() else {}
            if "wdks" in g.vertex_attributes(): weights_arg["vertex_weights"] = "wdks"
            partition = g.community_infomap(**weights_arg, trials=INFOMAP_TRIALS)
            communities = [[g.vs[idx]['name'] for idx in comm] for comm in partition]
            modularity = partition.modularity if hasattr(partition, 'modularity') else g.modularity(partition)
            codelength = partition.codelength if hasattr(partition, 'codelength') else float('nan')
            metrics = (modularity, codelength)
            # Log only to file
            logging.info(f"Snapshot {snapshot_index}: Infomap found {len(communities)} communities (Modularity: {modularity:.4f}, Codelength: {codelength:.4f}).", extra={'file_only': True})
        except Exception as e:
            logging.error(f"Snapshot {snapshot_index}: Infomap failed: {e}", exc_info=False)
            return [[v['name'] for v in g.vs]], (0.0, float('nan')), None, pd.DataFrame() # Fallback

        # Identify Key Nodes
        key_node_indices = []
        g.vs['is_key_node'] = False
        if 'wdks' not in g.vertex_attributes():
             logging.warning(f"Snapshot {snapshot_index}: 'wdks' missing. Skipping backbone.")
             return communities, metrics, None, pd.DataFrame()

        for comm_indices in partition:
            if not comm_indices: continue
            valid_comm_wdks = [(idx, g.vs[idx]['wdks']) for idx in comm_indices if np.isfinite(g.vs[idx]['wdks'])]
            if valid_comm_wdks:
                max_wdks_node_idx = max(valid_comm_wdks, key=lambda item: item[1])[0]
            elif comm_indices:
                max_wdks_node_idx = comm_indices[0]
            else: continue
            key_node_indices.append(max_wdks_node_idx)
            g.vs[max_wdks_node_idx]['is_key_node'] = True

        if not key_node_indices:
            logging.warning(f"Snapshot {snapshot_index}: No key nodes found. Skipping backbone.")
            return communities, metrics, None, pd.DataFrame()

        # Build Backbone Network (Simplified)
        backbone_node_indices = set(key_node_indices)
        if len(key_node_indices) > 1 and "evo_weight" in g.edge_attributes():
            weights_for_paths = [1.0 / (w + 1e-10) for w in g.es['evo_weight']]
            components = g.components(mode="weak")
            key_nodes_in_component = collections.defaultdict(list)
            for idx in key_node_indices: key_nodes_in_component[components.membership[idx]].append(idx)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Couldn't reach some vertices.*")
                for nodes_in_comp in key_nodes_in_component.values():
                    if len(nodes_in_comp) <= 1: continue
                    for source_idx in nodes_in_comp:
                        target_indices = [idx for idx in nodes_in_comp if idx != source_idx]
                        if not target_indices: continue
                        try:
                            paths = g.get_shortest_paths(source_idx, to=target_indices, weights=weights_for_paths, mode='out', output="vpath",algorithm="dijkstra")
                            for path_nodes in paths:
                                if path_nodes: backbone_node_indices.update(path_nodes)
                        except Exception: pass

        if not backbone_node_indices: return communities, metrics, None, pd.DataFrame()

        backbone_graph = g.subgraph(backbone_node_indices)
        if not backbone_graph or backbone_graph.vcount() == 0:
             return communities, metrics, None, pd.DataFrame()

        isolated_key_nodes_bb = [v.index for v in backbone_graph.vs if v['is_key_node'] and backbone_graph.degree(v.index) == 0]
        if isolated_key_nodes_bb: backbone_graph.delete_vertices(isolated_key_nodes_bb)

        # Log only to file
        logging.info(f"Snapshot {snapshot_index}: Backbone {backbone_graph.vcount()}v/{backbone_graph.ecount()}e", extra={'file_only': True})

        # Create Backbone DataFrame (Simplified)
        backbone_data = []
        df_cols = set(df.columns) if df is not None else set()
        bb_attrs = backbone_graph.vertex_attributes()

        for v_bb in backbone_graph.vs:
            node_name = v_bb['name']
            node_info = {'node_id': node_name}
            for attr in ['is_key_node', 'wdks', 'Date', 'Location', 'Lineage', 'Clade', 'Ancestor_ID'] + optional_attrs:
                 if attr in bb_attrs: node_info[attr] = v_bb[attr]
            if df is not None and node_name in df.index:
                 if 'ID' in df_cols and 'ID' not in node_info: node_info['ID'] = df.loc[node_name, 'ID']
                 if 'Ancestor' in df_cols: node_info['Ancestor'] = df.loc[node_name, 'Ancestor']
            backbone_data.append(node_info)

        backbone_df = pd.DataFrame(backbone_data)
        if not backbone_df.empty:
            col_order = ['node_id', 'ID', 'Date', 'is_key_node', 'wdks', 'Location', 'Lineage', 'Clade'] + \
                        [attr for attr in optional_attrs if attr in backbone_df.columns] + \
                        ['Ancestor', 'Ancestor_ID']
            final_cols = [col for col in col_order if col in backbone_df.columns] + \
                         [col for col in backbone_df.columns if col not in col_order]
            backbone_df = backbone_df[final_cols].set_index('node_id')

        return communities, metrics, backbone_graph, backbone_df

    def detect_communities_and_backbones(self,
                                         graphs_path: Optional[str] = None,
                                         raw_results_path: Optional[str] = None
                                         ) -> Tuple[str, str, str, str]:
        """Performs community detection and backbone extraction for all graphs in parallel."""
        logging.info("Starting community detection and backbone extraction...")
        start_detect_time = dt.datetime.now()

        graphs_to_process, graphs_path = self._load_or_get_data(
            graphs_path, self.temporal_graphs, "temporal graphs",
            "Temporal_graphs_{start}_to_{end}.h5"
        )
        # Load raw data only if needed and not already loaded
        mcan_dataframes = None
        if not self.raw_networks_data:
             try:
                 mcan_dataframes, _ = self._load_or_get_data(
                     raw_results_path, [], "raw McAN results", # Pass empty list to force load or find default
                     "McAN_raw_results_{start}_to_{end}.h5"
                 )
             except ValueError: # Handle case where raw data is truly optional/unavailable
                  logging.warning("Raw McAN results not found or provided. Backbone tables may lack some info.")
                  mcan_dataframes = [None] * len(graphs_to_process)
        else:
             mcan_dataframes = self.raw_networks_data


        if not isinstance(graphs_to_process, list) or not graphs_to_process:
            logging.warning("Graph list empty. Skipping detection.")
            # Return empty file paths
            suffix = f"{self.start_date}_to_{self.end_date}.h5" if self.start_date and self.end_date else "empty.h5"
            paths = [os.path.join(self.output_path, f"{prefix}_{suffix}") for prefix in
                     ["Community_structures", "Network_metrics", "Backbone_networks", "Backbone_tables"]]
            for p in paths:
                IOManager.save_to_hdf5([], p)  # Save empty lists
            return tuple(paths)

        num_graphs = len(graphs_to_process)
        dfs_aligned = [(mcan_dataframes[i] if mcan_dataframes and i < len(mcan_dataframes) else None) for i in
                       range(num_graphs)]

        tasks = [(graph, i, dfs_aligned[i], self.output_path, self.optional_attrs)
                 for i, graph in enumerate(graphs_to_process)]
        num_tasks = len(tasks)
        logging.info(f"Detecting communities for {num_tasks} graphs using {self.num_processes} processes.")

        detection_results_map = {}
        try:
            with closing(ParallelProcessor.get_process_pool(processes=self.num_processes, maxtasksperchild=1)) as pool:
                processed_count = 0
                last_milestone = 0
                for index, result_tuple in pool.imap_unordered(TempSnap._detect_partition_task, tasks,
                                                               chunksize=max(1, num_tasks // (self.num_processes * 2))):
                    detection_results_map[index] = result_tuple
                    processed_count += 1
                    current_percent = (processed_count * 100) // num_tasks
                    current_milestone = (current_percent // 20) * 20
                    if current_milestone > last_milestone or processed_count == num_tasks:
                        logging.info(f"Community detection progress: {current_milestone}% ({processed_count}/{num_tasks})")
                        last_milestone = current_milestone
        except Exception as e:
            logging.error(f"Parallel community detection failed: {e}", exc_info=True)
            raise

        # Collect results
        results = [detection_results_map.get(i, ([], (0.0, float('nan')), None, pd.DataFrame())) for i in range(num_graphs)]
        final_communities = [r[0] for r in results]
        final_metrics = [r[1] for r in results]
        final_backbones = [r[2] for r in results]
        final_backbone_dfs = [r[3] for r in results]

        logging.info(f"Finished detection in {(dt.datetime.now() - start_detect_time).total_seconds():.2f}s.")

        # Save results
        suffix = f"{self.start_date}_to_{self.end_date}.h5" if self.start_date and self.end_date else "results.h5"
        paths = {}
        results_to_save = {
            "Community_structures": final_communities,
            "Network_metrics": final_metrics,
            "Backbone_networks": final_backbones,
            "Backbone_tables": final_backbone_dfs
        }
        for prefix, data in results_to_save.items():
            filepath = os.path.join(self.output_path, f"{prefix}_{suffix}")
            try:
                # Pass self.num_processes to save_to_hdf5
                IOManager.save_to_hdf5(data, filepath, num_threads=self.num_processes)
                paths[prefix] = filepath
            except Exception as e:
                 logging.error(f"Failed to save {prefix}: {e}", exc_info=True)
                 paths[prefix] = None # Indicate save failure

        return paths.get("Community_structures"), paths.get("Network_metrics"), paths.get("Backbone_networks"), paths.get("Backbone_tables")

    def run_full_pipeline(self, n_rows_mcan: Optional[int] = None) -> Tuple[str, str, str, str]:
        """Runs the complete analysis pipeline."""
        logging.info("--- Starting Full TempSnap Pipeline ---")
        pipeline_start_time = dt.datetime.now()

        try:
            # Step 1: McAN Simulation
            raw_results_filepath = self.run_mcan_simulation(n_rows=n_rows_mcan)

            # Step 2: Graph Building
            graphs_filepath = self.build_temporal_graphs(raw_results_path=raw_results_filepath)

            # Step 3: Community Detection & Backbone Extraction
            comm_path, met_path, bb_path, bbt_path = self.detect_communities_and_backbones(
                graphs_path=graphs_filepath,
                raw_results_path=raw_results_filepath
            )

            logging.info(f"--- Pipeline completed successfully in {(dt.datetime.now() - pipeline_start_time).total_seconds():.2f}s ---")
            return comm_path, met_path, bb_path, bbt_path

        except Exception as e:
            logging.critical(f"Pipeline failed: {e}", exc_info=True)
            raise

    def analyze_temporal_distribution(self,
                                    category_column: str = 'Lineage',
                                    location_column: Optional[str] = None,
                                    location_filter: Optional[str] = None,
                                    raw_results_path: Optional[str] = None,
                                    graphs_list: Optional[List] = None,
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None,
                                    time_interval: Optional[int] = None,
                                    save_plot: bool = True,
                                    plot_format: str = 'svg',
                                    dpi: int = 300,
                                    figsize: Tuple[int, int] = (12, 9),
                                    color_palette: str = 'Set3',
                                    max_categories: Optional[int] = None,
                                    legend_ncol: int = 4,
                                    font_path: Optional[str] = None,
                                    custom_title: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """
        Analyzes and visualizes the temporal distribution of categories (e.g., Lineage) across time snapshots.
        
        Parameters:
        -----------
        category_column : str
            Column name to analyze distribution for (default: 'Lineage')
        location_column : str, optional
            Column name for location filtering (e.g., 'Location', 'Country')
        location_filter : str or list, optional
            Specific location/country to filter by (e.g., 'Brazil') or list of locations (e.g., ['Brazil', 'USA'])
        raw_results_path : str, optional
            Path to raw McAN results HDF5 file
        graphs_list : List, optional
            Pre-loaded list of graph objects (alternative to raw_results_path or temporal_graphs)
        start_date : str, optional
            Start date for analysis range (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for analysis range (format: 'YYYY-MM-DD')
        time_interval : int, optional
            Time interval in days for date estimation when missing
        save_plot : bool
            Whether to save the plot to file
        plot_format : str
            Format for saved plot ('png', 'pdf', 'svg')
        dpi : int
            Resolution for saved plot
        figsize : tuple
            Figure size (width, height) in inches
        color_palette : str
            Color palette for the plot
        max_categories : int, optional
            Maximum number of categories to display. Rest will be grouped as 'Others'
        legend_ncol : int
            Number of columns in the legend (default: 4)
        font_path : str, optional
            Path to custom font file (e.g., '/path/to/font.ttf')
        custom_title : str, optional
            Custom title for the plot (overrides auto-generated title)
            
        Returns:
        --------
        Tuple[pd.DataFrame, str]
            Statistics DataFrame and plot file path
        """
        logging.info(f"Starting temporal distribution analysis for '{category_column}'")
        
        # Load graph data - prioritize graphs_list if provided
        if graphs_list is not None:
            source_graphs = graphs_list
            logging.info(f"Using provided graph list with {len(source_graphs)} snapshots")
        elif hasattr(self, 'temporal_graphs') and self.temporal_graphs:
            source_graphs = self.temporal_graphs
            logging.info(f"Using instance temporal graphs with {len(source_graphs)} snapshots")
        else:
            # Try to load graphs from file path
            try:
                source_graphs, _ = self._load_or_get_data(
                    None, [], "temporal graphs",
                    "Temporal_graphs_{start}_to_{end}.h5"
                )
            except:
                raise ValueError("No graph data available for analysis. Please provide graphs_list or ensure temporal_graphs are available.")
        
        if not isinstance(source_graphs, list) or not source_graphs:
            raise ValueError("No temporal graph data available for analysis")
        
        # Determine date range and filter snapshots using Trace module
        from Trace import extract_date_range
        start_idx, end_idx, actual_max_dates = extract_date_range(
            source_graphs, start_date, end_date, time_interval or 1
        )
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No data available in the specified date range")
        
        # Filter graphs based on date range
        filtered_graphs = source_graphs[start_idx:end_idx + 1]
        if actual_max_dates:
            filtered_dates = actual_max_dates
        else:
            filtered_dates = [None] * len(filtered_graphs)
        
        logging.info(f"Analyzing snapshots {start_idx} to {end_idx} (total: {len(filtered_graphs)} snapshots)")
        
        # Extract time points and prepare data from graph objects
        temporal_stats = []
        valid_snapshots = []
        
        for i, graph in enumerate(filtered_graphs):
            if graph is None or not hasattr(graph, 'vs') or graph.vcount() == 0:
                continue
                
            # Get the time point for this snapshot
            if filtered_dates and i < len(filtered_dates) and filtered_dates[i]:
                time_point = filtered_dates[i]
            else:
                # Fallback: extract from graph vertex Date attributes
                try:
                    if 'Date' in graph.vs.attributes():
                        date_values = [v['Date'] for v in graph.vs if 'Date' in v.attributes() and v['Date']]
                        if date_values:
                            time_point = pd.to_datetime(date_values).max()
                        else:
                            time_point = start_idx + i
                    else:
                        time_point = start_idx + i
                except Exception as e:
                    logging.warning(f"Snapshot {i}: Could not parse dates from graph: {e}, using index {start_idx + i}")
                    time_point = start_idx + i
            
            # Extract node data from graph
            node_data = []
            if not graph.vs:
                continue
                
            for v in graph.vs:
                node_info = {}
                for attr in v.attributes():
                    node_info[attr] = v[attr]
                node_data.append(node_info)
            
            if not node_data:
                continue
                
            # Convert to DataFrame for easier processing
            graph_df = pd.DataFrame(node_data)
            
            # Apply location filter if specified
            filtered_df = graph_df.copy()
            if location_column and location_filter:
                if location_column not in graph_df.columns:
                    logging.warning(f"Snapshot {i}: Column '{location_column}' not found in graph, skipping location filter")
                else:
                    # Convert single location to list for uniform processing
                    if isinstance(location_filter, str):
                        location_list = [location_filter]
                    else:
                        location_list = list(location_filter)
                    
                    # Handle hierarchical location data (e.g., "South America / Brazil / Rio de Janeiro / Rio de Janeiro")
                    if location_column in ['Location', 'Country']:
                        def extract_country(location_str):
                            if pd.isna(location_str) or not isinstance(location_str, str):
                                return None
                            parts = [part.strip() for part in str(location_str).split('/')]
                            if len(parts) >= 2:
                                return parts[1]  # Take the second part (country)
                            elif len(parts) == 1:
                                return parts[0]
                            return None
                        
                        filtered_df['_country_extracted'] = filtered_df[location_column].apply(extract_country)
                        filtered_df = filtered_df[filtered_df['_country_extracted'].isin(location_list)]
                    else:
                        # Direct filtering for other location columns
                        filtered_df = filtered_df[filtered_df[location_column].isin(location_list)]
                    
                    # Log filtering result
                    original_count = len(graph_df)
                    filtered_count = len(filtered_df)
                    logging.info(f"Snapshot {i}: Filtered from {original_count} to {filtered_count} samples for location(s): {location_list}", extra={'file_only': True})
            
            # Check if category column exists
            if category_column not in filtered_df.columns:
                logging.warning(f"Snapshot {i}: Column '{category_column}' not found in graph, skipping")
                continue
            
            # Calculate proportions for filtered data only
            category_counts = filtered_df[category_column].value_counts()
            total_count = len(filtered_df)
            
            if total_count == 0:
                logging.warning(f"Snapshot {i}: No data after location filtering")
                continue
                
            category_proportions = (category_counts / total_count * 100).round(2)
            
            # Store results
            for category, proportion in category_proportions.items():
                temporal_stats.append({
                    'Time_Point': time_point,
                    'Snapshot_Index': start_idx + i,  # Use absolute index
                    category_column: category,
                    'Count': category_counts[category],
                    'Total_Count': total_count,
                    'Proportion_Percent': proportion
                })
            
            valid_snapshots.append(start_idx + i)  # Use absolute index
        
        if not temporal_stats:
            raise ValueError("No valid data found for analysis")
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame(temporal_stats)
        
        # Sort by time point
        stats_df = stats_df.sort_values('Time_Point').reset_index(drop=True)
        
        # Create pivot table for plotting
        pivot_df = stats_df.pivot_table(
            index='Time_Point', 
            columns=category_column, 
            values='Proportion_Percent', 
            fill_value=0
        )
        
        # Handle too many categories by grouping less frequent ones as "Others"
        if max_categories and len(pivot_df.columns) > max_categories:
            # Calculate total counts for each category
            category_totals = stats_df.groupby(category_column)['Count'].sum().sort_values(ascending=False)
            top_categories = category_totals.head(max_categories - 1).index.tolist()
            
            # Create new pivot table with "Others" category
            pivot_df_reduced = pivot_df[top_categories].copy()
            others_sum = pivot_df.drop(columns=top_categories).sum(axis=1)
            pivot_df_reduced['Others'] = others_sum
            pivot_df = pivot_df_reduced
            
            logging.info(f"Grouped {len(category_totals) - max_categories + 1} categories into 'Others'")
        
        # Set up the plot with academic style
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                plt.style.use('default')
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Set up custom font if provided
        font_prop_title = None
        font_prop_label = None
        if font_path:
            try:
                import matplotlib.font_manager as font_manager
                font_manager.fontManager.addfont(font_path)
                # Create separate font properties with specific sizes
                font_prop_title = font_manager.FontProperties(fname=font_path, size=26, weight='bold')
                font_prop_label = font_manager.FontProperties(fname=font_path, size=20, weight='bold')
                logging.info(f"Custom font loaded successfully: {font_path}")

            except Exception as e:
                logging.warning(f"Failed to load custom font: {e}")
                font_prop_title = None
                font_prop_label = None
        
        # Get colors for categories
        categories = pivot_df.columns.tolist()
        
        # Generate enough distinct colors
        def generate_colors(n_colors, palette_name=color_palette):
            if n_colors <= 20:
                # Use standard colormap
                try:
                    colors = plt.cm.get_cmap(palette_name)(np.linspace(0, 1, n_colors))
                except ValueError:
                    # Fallback to tab20 if palette not found
                    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_colors))
            else:
                # For many colors, combine multiple palettes
                base_palettes = ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
                colors = []
                palette_idx = 0
                colors_per_palette = 20
                
                for i in range(n_colors):
                    if i > 0 and i % colors_per_palette == 0:
                        palette_idx += 1
                    
                    current_palette = base_palettes[palette_idx % len(base_palettes)]
                    color_idx = i % colors_per_palette
                    
                    try:
                        cmap = plt.cm.get_cmap(current_palette)
                        # Adjust number of colors if palette has fewer colors
                        n_available = cmap.N if hasattr(cmap, 'N') else 256
                        if n_available < colors_per_palette:
                            colors_per_palette = n_available
                        color = cmap(color_idx / max(1, colors_per_palette - 1))
                    except:
                        # Generate random color as fallback
                        np.random.seed(i)  # For reproducible colors
                        color = np.random.rand(3).tolist() + [1.0]  # RGB + alpha
                    
                    colors.append(color)
                
                colors = np.array(colors)
            
            return colors
        
        colors = generate_colors(len(categories))
        
        # Create stacked bar plot
        bottom = np.zeros(len(pivot_df))
        bars = []
        
        for i, category in enumerate(categories):
            values = pivot_df[category].values
            bar = ax.bar(range(len(pivot_df)), values, bottom=bottom, 
                        color=colors[i], label=category, alpha=0.8, 
                        edgecolor='white', linewidth=0.5)
            bars.append(bar)
            bottom += values
        
        # Customize the plot
        if font_prop_title and font_prop_label:
            # Use custom font with pre-set sizes
            ax.set_xlabel('Time Point', fontproperties=font_prop_label)
            ax.set_ylabel(f'{category_column} Proportion (%)', fontproperties=font_prop_label)
        else:
            # Use default font with manual size settings
            ax.set_xlabel('Time Point', fontsize=20, fontweight='bold')
            ax.set_ylabel(f'{category_column} Proportion (%)', fontsize=20, fontweight='bold')
        
        # Set title
        if custom_title:
            title = custom_title
        else:
            title = f'Temporal Distribution of {category_column}'
            if location_filter:
                if isinstance(location_filter, list):
                    if len(location_filter) <= 3:
                        location_str = ', '.join(location_filter)
                    else:
                        location_str = f"{', '.join(location_filter[:2])} and {len(location_filter)-2} others"
                else:
                    location_str = location_filter
                title += f' in {location_str}'
        # Set title with appropriate font
        if font_prop_title:
            ax.set_title(title, fontproperties=font_prop_title, pad=10)
        else:
            ax.set_title(title, fontsize=26, fontweight='bold', pad=10)

        # Format x-axis
        time_labels = []
        for time_point in pivot_df.index:
            if isinstance(time_point, (pd.Timestamp, dt.datetime)):
                time_labels.append(time_point.strftime('%Y-%m-%d'))
            else:
                time_labels.append(str(time_point))
        
        ax.set_xticks(range(len(pivot_df)))
        ax.set_xticklabels(time_labels, rotation=0, ha='center')  # 不倾斜显示
        
        # Set tick label font if custom font is available
        if font_prop_label:
            try:
                import matplotlib.font_manager as font_manager
                font_prop_tick = font_manager.FontProperties(fname=font_path, size=10)
                for label in ax.get_xticklabels():
                    label.set_fontproperties(font_prop_tick)
                for label in ax.get_yticklabels():
                    label.set_fontproperties(font_prop_tick)
            except:
                pass  # Use default if font setting fails
        else:
            # Set tick label size for default font
            ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Format y-axis
        ax.set_ylim(0, 100)
        
# Add legend below the plot
        legend_kwargs = {
            'bbox_to_anchor': (0.5, -0.1), 
            'loc': 'upper center', 
            'fontsize': 12, 
            'ncol': legend_ncol, 
            'frameon': True, 
            'fancybox': True, 
            'shadow': True,'markerscale': 1.5 
        }
        
        # Apply custom font to legend if available
        if font_prop_label:
            # For legend, we need to create a smaller font
            try:
                import matplotlib.font_manager as font_manager
                font_prop_legend = font_manager.FontProperties(fname=font_path, size=12)
                legend_kwargs['prop'] = font_prop_legend
            except:
                pass  # Use default if font setting fails
        
        ax.legend(**legend_kwargs)
        
        # Grid styling
        ax.grid(False)

        
        # Adjust layout to accommodate legend below
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2 + 0.05 * ((len(categories) // legend_ncol) + 1))
        
        # Save plot if requested
        plot_path = None
        if save_plot:
            # Create filename
            filter_suffix = ""
            if location_filter:
                if isinstance(location_filter, list):
                    if len(location_filter) <= 3:
                        filter_suffix = f"_{'_'.join(location_filter)}"
                    else:
                        filter_suffix = f"_{len(location_filter)}regions"
                else:
                    filter_suffix = f"_{location_filter}"
            filename = f"temporal_distribution_{category_column}{filter_suffix}.{plot_format}"
            plot_path = os.path.join(self.output_path, filename)
            
            try:
                plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                logging.info(f"Plot saved to: {plot_path}")
            except Exception as e:
                logging.error(f"Failed to save plot: {e}")
                plot_path = None
        
        # Save statistics DataFrame
        stats_filename = f"temporal_distribution_stats_{category_column}"
        if location_filter:
            if isinstance(location_filter, list):
                if len(location_filter) <= 3:
                    stats_filename += f"_{'_'.join(location_filter)}"
                else:
                    stats_filename += f"_{len(location_filter)}regions"
            else:
                stats_filename += f"_{location_filter}"
        stats_filename += ".csv"
        stats_path = os.path.join(self.output_path, stats_filename)
        
        try:
            stats_df.to_csv(stats_path, index=False)
            logging.info(f"Statistics saved to: {stats_path}")
        except Exception as e:
            logging.error(f"Failed to save statistics: {e}")
        
        # Show plot
        plt.show()
        
        # Log summary
        logging.info(f"Analysis completed for {len(valid_snapshots)} time snapshots")
        logging.info(f"Found {len(categories)} unique {category_column} categories")
        if location_filter:
            total_samples = stats_df['Total_Count'].sum()
            if isinstance(location_filter, list):
                location_str = f"{len(location_filter)} regions"
            else:
                location_str = location_filter
            logging.info(f"Total samples in {location_str}: {total_samples}")
            return stats_df, plot_path or ""

    def load_temporal_graphs(self, graphs_path: Optional[str] = None) -> List:
        """Loads temporal graphs from HDF5 file or returns existing graphs."""
        if graphs_path:
            logging.info(f"Loading temporal graphs from: {graphs_path}")
            return IOManager.load_from_hdf5(graphs_path, parallelism_level=self.num_processes)
        elif self.temporal_graphs:
            logging.info("Using existing temporal graphs from instance.")
            return self.temporal_graphs
        else:
            # Try default path
            default_path = os.path.join(self.output_path, f"Temporal_graphs_{self.start_date}_to_{self.end_date}.h5")
            if os.path.exists(default_path):
                logging.info(f"Loading temporal graphs from default path: {default_path}")
                return IOManager.load_from_hdf5(default_path, parallelism_level=self.num_processes)
            else:
                raise ValueError("No temporal graphs available. Please provide graphs_path or ensure temporal graphs exist.")

# Example Usage (Simplified)
if __name__ == "__main__":
    OUTPUT_DIRECTORY = "./temp_snap_output_simplified"
    SAMPLES_FILE = "/path/to/your/samples.tsv" # IMPORTANT: Replace with actual path
    START_DATE = "2023-01-01"
    END_DATE = "2023-01-05" # Shorter range for faster testing
    TIME_INTERVAL = dt.timedelta(days=1)
    NUM_PROCESSES = 4
    OPTIONAL_ATTRIBUTES = ['Lineage', 'Clade']

    # Setup logging for the main script execution
    LogManager.configure_logging(OUTPUT_DIRECTORY, process_type="main")

    try:
        analyzer = TempSnap(
            output_path=OUTPUT_DIRECTORY,
            samples_path=SAMPLES_FILE,
            start_date=START_DATE,
            end_date=END_DATE,
            time_interval=TIME_INTERVAL,
            num_processes=NUM_PROCESSES,
            optional_attrs=OPTIONAL_ATTRIBUTES
        )

        # Run pipeline (limit rows for testing)
        results = analyzer.run_full_pipeline(n_rows_mcan=500)

        # Example: Analyze temporal distribution of Lineage
        logging.info("--- Running Temporal Distribution Analysis ---")
        
        # Analyze overall lineage distribution over time using temporal graphs
        stats_df, plot_path = analyzer.analyze_temporal_distribution(
            category_column='Lineage',
            graphs_list=analyzer.temporal_graphs,  # Use the built temporal graphs
            save_plot=True,
            plot_format='png'
        )
        logging.info(f"Overall lineage statistics shape: {stats_df.shape}")
        
        # Analyze lineage distribution for a specific country (example)
        # stats_df_country, plot_path_country = analyzer.analyze_temporal_distribution(
        #     category_column='Lineage',
        #     graphs_list=analyzer.temporal_graphs,
        #     location_column='Location',
        #     location_filter='Brazil',  # Replace with actual country in your data
        #     save_plot=True,
        #     plot_format='png'
        # )
        
        # Example with other category columns
        # stats_df_clade, _ = analyzer.analyze_temporal_distribution(
        #     category_column='Clade',
        #     graphs_list=analyzer.temporal_graphs,
        #     save_plot=True
        # )

        logging.info("--- TempSnap Example Finished ---")
        logging.info(f"Results saved in: {OUTPUT_DIRECTORY}")
        # logging.info(f"Communities: {results[0]}") # Optional: Log file paths

    except FileNotFoundError:
        logging.error(f"Error: Samples file not found at '{SAMPLES_FILE}'. Please provide the correct path.")
    except ImportError:
         logging.error("Error: McAN module not found.")
    except Exception as main_e:
        logging.critical(f"Example execution failed: {main_e}", exc_info=True)