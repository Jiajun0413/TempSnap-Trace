from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import os
from TempSnap import IOManager
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
import torch

# ================ Data Models ================
@dataclass
class TrackingNode:
    """Represents a node during the tracking process."""
    date: pd.Timestamp
    community_id: int
    lineage_counts: Dict[str, int]
    similarity: float
    matched_from: Optional[Tuple[int, int]] # (time_index, community_index)
    contains_label: bool
    core_node: str
    core_node_mutations: str
    alternative_matched_from: List[Tuple[int, int]] = field(default_factory=list) # List of (time_index, community_index)

@dataclass
class CommunityData:
    """Lightweight representation of community data."""
    id: int
    members: Set[str] = field(default_factory=set)
    mutations: Set[str] = field(default_factory=set) # Set of mutation strings for members
    has_target: bool = False
    lineage_counts: Dict[str, int] = field(default_factory=dict)
    core_node: str = ""
    core_mutations: str = ""

    def extract_mutations_dict(self) -> Dict[str, int]:
        """Extracts community mutations into a dictionary {mutation: count}."""
        result = {}
        for mutation_str in self.mutations:
            if mutation_str:
                for item in mutation_str.split(';'):
                    item = item.strip()
                    if item:
                        result[item] = result.get(item, 0) + 1
        return result

# ================ Similarity Calculation ================
@torch.jit.script
def compute_node_similarity_matrix(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Vectorized Jaccard similarity calculation using PyTorch."""
    batch_size, n_mutations = vec1.size()
    n_nodes2 = vec2.size(0)

    vec1_sums = vec1.sum(dim=1)  # [batch_size]
    vec2_sums = vec2.sum(dim=1)  # [n_nodes2]

    similarity_matrix = torch.zeros((batch_size, n_nodes2), device=vec1.device)

    # Process in batches to manage memory
    batch_inner = 128
    for i in range(0, batch_size, batch_inner):
        end_i = min(i + batch_inner, batch_size)
        batch_vec1 = vec1[i:end_i]
        batch_sums1 = vec1_sums[i:end_i].unsqueeze(1)

        intersections = torch.mm(batch_vec1, vec2.t())
        unions = batch_sums1 + vec2_sums - intersections

        mask = unions > 0
        similarity_matrix[i:end_i] = torch.where(mask, intersections / unions, torch.zeros_like(intersections))

    return similarity_matrix

class SimilarityCalculator(ABC):
    """Base class for similarity calculation."""
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.call_count = 0
        self.mutation_cache = {} # Cache for parsed mutation strings

    def extract_mutations_dict(self, mutation_set: Set[str]) -> Dict[str, int]:
        """Extracts mutations from a set of strings into a count dictionary."""
        result = {}
        for mutation_str in mutation_set:
            if not mutation_str:
                continue
            for item in mutation_str.split(';'):
                item = item.strip()
                if item:
                    result[item] = result.get(item, 0) + 1
        return result

    def extract_mutations_with_cache(self, mutations_set: Set[str], use_cache=True) -> List[Dict[str, int]]:
        """Extracts mutations with optional caching."""
        if not use_cache:
            # Return a list containing a single aggregated dictionary if no cache
            return [self.extract_mutations_dict(mutations_set)]

        result = []
        for mutation_str in mutations_set:
            if not mutation_str:
                continue

            if mutation_str not in self.mutation_cache:
                mutations = {}
                for item in mutation_str.split(';'):
                    item = item.strip()
                    if item:
                        mutations[item] = mutations.get(item, 0) + 1
                self.mutation_cache[mutation_str] = mutations

            if self.mutation_cache[mutation_str]: # Only add if non-empty
                result.append(self.mutation_cache[mutation_str])

        # Simple cache management
        self.call_count += 1
        if self.call_count % 1000 == 0 and len(self.mutation_cache) > 10000:
            # print("Clearing mutation cache...") # Optional: for debugging
            self.mutation_cache.clear()

        return result # Returns list of dicts, one per member mutation string

    @abstractmethod
    def calculate_similarity(self, mutations1: Set[str], mutations2: Set[str]) -> float:
        """Core calculation method to be implemented by subclasses."""
        pass

    def calculate(self, comm1: CommunityData, comm2: CommunityData) -> float:
        """Unified interface for calculating similarity between two communities."""
        if not comm1.mutations or not comm2.mutations:
            return 0.0
        # Optimization: If mutation sets are identical (e.g., comparing a community to itself implicitly)
        if comm1.mutations is comm2.mutations:
             return 1.0

        return self.calculate_similarity(comm1.mutations, comm2.mutations)

    def batch_calculate(self, batch_args: Tuple[Tuple[List[List[CommunityData]], float], List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int, float]]:
        """Framework for batch similarity calculation."""
        (preprocessed_communities, threshold), community_pairs = batch_args

        community_cache = self._preload_communities(preprocessed_communities, community_pairs)
        return self._process_batch(community_pairs, community_cache, threshold)

    def _preload_communities(self, preprocessed_communities: List[List[CommunityData]], community_pairs: List[Tuple[int, int, int, int]]) -> Dict[Tuple[int, int], CommunityData]:
        """Preloads required communities into a cache."""
        community_cache = {}
        all_indices = set()
        for t_idx, c_idx, prev_t_idx, prev_c_idx in community_pairs:
            all_indices.add((t_idx, c_idx))
            all_indices.add((prev_t_idx, prev_c_idx))

        for t, c in all_indices:
            if t < len(preprocessed_communities) and c < len(preprocessed_communities[t]):
                 community_cache[(t, c)] = preprocessed_communities[t][c]

        return community_cache

    def _process_batch(self, community_pairs: List[Tuple[int, int, int, int]], community_cache: Dict[Tuple[int, int], CommunityData], threshold: float) -> List[Tuple[int, int, int, int, float]]:
        """Processes a batch of community pairs."""
        results = []
        for t_idx, c_idx, prev_t_idx, prev_c_idx in community_pairs:
            comm1 = community_cache.get((t_idx, c_idx))
            comm2 = community_cache.get((prev_t_idx, prev_c_idx))

            if comm1 and comm2:
                sim = self.calculate(comm1, comm2)
                if sim > threshold:
                    results.append((t_idx, c_idx, prev_t_idx, prev_c_idx, sim))
        return results

class GPUSimilarityCalculator(SimilarityCalculator):
    """GPU-accelerated similarity calculation (Corrected Logic)."""

    def calculate_similarity(self, mutations1: Set[str], mutations2: Set[str]) -> float:
        """Calculates similarity using GPU based on member-wise comparison."""
        # Prepare vectors returns matrices where rows are members
        vectors1, vectors2, _ = self._prepare_vectors(mutations1, mutations2)
        # Check if vectors could be created and are not empty
        if vectors1 is None or vectors2 is None or vectors1.shape[0] == 0 or vectors2.shape[0] == 0:
            return 0.0

        return self._compute_gpu_similarity(vectors1, vectors2)

    def _prepare_vectors(self, mutations1: Set[str], mutations2: Set[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
        """Prepares member mutation vectors for GPU computation (Original Logic)."""
        # extract_mutations_with_cache returns a list of dicts, one per member
        nodes1_dicts = self.extract_mutations_with_cache(mutations1)
        nodes2_dicts = self.extract_mutations_with_cache(mutations2)

        if not nodes1_dicts or not nodes2_dicts:
            return None, None, None # Cannot compare if one community has no members with mutations

        # Build common mutation space from all members in both communities
        all_mutations_set = set()
        for node_dict in nodes1_dicts + nodes2_dicts:
            all_mutations_set.update(node_dict.keys())

        if not all_mutations_set:
             return None, None, None # No common mutations found

        all_mutations_list = sorted(list(all_mutations_set))
        mutation_to_idx = {m: i for i, m in enumerate(all_mutations_list)}
        vector_size = len(mutation_to_idx)

        # Create vector matrices: rows = members, cols = mutations
        vectors1 = np.zeros((len(nodes1_dicts), vector_size), dtype=np.float32)
        for i, node_dict in enumerate(nodes1_dicts):
            for mutation, value in node_dict.items():
                idx = mutation_to_idx.get(mutation)
                if idx is not None:
                    vectors1[i, idx] = value

        vectors2 = np.zeros((len(nodes2_dicts), vector_size), dtype=np.float32)
        for i, node_dict in enumerate(nodes2_dicts):
            for mutation, value in node_dict.items():
                idx = mutation_to_idx.get(mutation)
                if idx is not None:
                    vectors2[i, idx] = value

        return vectors1, vectors2, vector_size


    def _compute_gpu_similarity(self, vectors1: np.ndarray, vectors2: np.ndarray) -> float:
        """Computes similarity on GPU using member-wise comparison (Original Logic)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Declare tensors outside try block for finally clause
        torch_vectors1 = None
        torch_vectors2 = None
        similarity_matrix = None
        try:
            torch_vectors1 = torch.tensor(vectors1, device=device) # Shape: [n_members1, n_mutations]
            torch_vectors2 = torch.tensor(vectors2, device=device) # Shape: [n_members2, n_mutations]

            # Calculate pairwise similarity between all members
            # compute_node_similarity_matrix handles batching internally if needed
            similarity_matrix = compute_node_similarity_matrix(torch_vectors1, torch_vectors2) # Shape: [n_members1, n_members2]

            # Calculate bidirectional similarity from the member-wise matrix
            similarity = self._compute_bidirectional_similarity(similarity_matrix)

            # Optional: Clear cache periodically
            if self.call_count % 100 == 0:
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()

            return similarity
        except Exception as e:
            print(f"Error during GPU similarity computation: {e}")
            return 0.0 # Fallback on error
        finally:
            # Ensure tensors are deleted to free GPU memory
            del torch_vectors1
            del torch_vectors2
            del similarity_matrix

    def _compute_bidirectional_similarity(self, similarity_matrix: torch.Tensor) -> float:
        """Calculates harmonic mean of forward/backward average best matches (Original Logic)."""
        # similarity_matrix shape: [n_members1, n_members2]

        if similarity_matrix.numel() == 0: # Handle empty matrix case
             return 0.0

        # Forward match: Average of max similarity for each member in set 1 to set 2
        row_max_values, _ = torch.max(similarity_matrix, dim=1) # Max along dim 1 (columns) -> shape [n_members1]
        forward_match = torch.mean(row_max_values).item() if row_max_values.numel() > 0 else 0.0

        # Backward match: Average of max similarity for each member in set 2 to set 1
        col_max_values, _ = torch.max(similarity_matrix, dim=0) # Max along dim 0 (rows) -> shape [n_members2]
        backward_match = torch.mean(col_max_values).item() if col_max_values.numel() > 0 else 0.0

        # Harmonic mean
        if forward_match + backward_match > 1e-9: # Use tolerance for floating point
            return 2 * forward_match * backward_match / (forward_match + backward_match)
        else:
            return 0.0



def create_calculator(config: 'TrackingConfig') -> SimilarityCalculator:
    """Creates GPU similarity calculator."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU similarity calculator requires CUDA.")
    
    try:
        # Test if GPU is actually working
        _ = torch.tensor([1.0], device="cuda")
        print("Using GPU similarity calculator.")
        return GPUSimilarityCalculator(config.similarity_threshold)
    except Exception as e:
        raise RuntimeError(f"GPU calculator initialization failed: {e}")

# ================ Configuration & Utility Functions ================
@dataclass
class TrackingConfig:
    """Configuration for the tracking process."""
    similarity_threshold: float = 0.4
    time_window: int = 2
    use_gpu: bool = True
    n_processes: int = 4
    size_filter_threshold: float = 0.3 # Min size ratio for considering pairs
    weight_attr: str = "wdks" # Attribute for finding core node

def extract_date_range(extended_graphs: List, start_date: Optional[str], end_date: Optional[str], time_interval: int) -> Tuple[int, int, Optional[List[pd.Timestamp]]]:
    """Determines the start and end indices and actual dates based on graph data and specified range."""
    user_specified_start = start_date is not None
    user_specified_end = end_date is not None

    graph_dates = []
    for idx, g in enumerate(extended_graphs):
        max_date = None
        if g and hasattr(g, 'vs') and 'Date' in g.vs.attributes():
            valid_dates = [pd.Timestamp(v['Date']) for v in g.vs if 'Date' in v.attributes() and v['Date']]
            if valid_dates:
                max_date = max(valid_dates)
        graph_dates.append((idx, max_date))

    valid_graph_dates = [(idx, date) for idx, date in graph_dates if date is not None]
    if not valid_graph_dates:
        print("Warning: No valid dates found in graph data.")
        return -1, -1, None

    start_timestamp = pd.Timestamp(start_date) if user_specified_start else None
    end_timestamp = pd.Timestamp(end_date) if user_specified_end else None

    # Determine start index
    start_idx = 0
    if user_specified_start:
        # Find the latest graph index whose max date is <= start_timestamp
        candidates = [(idx, date) for idx, date in valid_graph_dates if date <= start_timestamp]
        if candidates:
            start_idx = max(candidates, key=lambda x: x[1])[0]
        # else: start_idx remains 0 if no graph date is <= start_timestamp

    # Determine end index
    end_idx = len(extended_graphs) - 1
    if user_specified_end:
        # Find the earliest graph index whose max date is >= end_timestamp
        candidates = [(idx, date) for idx, date in valid_graph_dates if date >= end_timestamp]
        if candidates:
            end_idx = min(candidates, key=lambda x: x[1])[0]
        else:
            # If no graph date >= end_date, use the last valid graph index
             end_idx = valid_graph_dates[-1][0]
    # else: end_idx remains len(extended_graphs) - 1

    if start_idx > end_idx:
        print(f"Warning: Invalid date range - start index ({start_idx}) is greater than end index ({end_idx}). Check start/end dates.")
        return -1, -1, None

    # Get actual max dates for the selected range [start_idx, end_idx]
    actual_max_dates = []
    last_valid_date = None
    for idx in range(start_idx, end_idx + 1):
        if idx < len(graph_dates) and graph_dates[idx][1] is not None:
            actual_max_dates.append(graph_dates[idx][1])
            last_valid_date = graph_dates[idx][1]
        elif last_valid_date is not None:
             # If current graph has no date, estimate based on previous + interval
            actual_max_dates.append(last_valid_date + pd.Timedelta(days=time_interval))
            last_valid_date = actual_max_dates[-1]
        else:
            # Fallback if the very first selected graph has no date
            fallback_date = pd.Timestamp(start_date) if start_date else pd.Timestamp.now().normalize()
            actual_max_dates.append(fallback_date)
            last_valid_date = fallback_date


    selected_dates_in_range = [date for idx, date in valid_graph_dates if start_idx <= idx <= end_idx]
    if selected_dates_in_range:
        print(f"Processing graphs from index {start_idx} to {end_idx}, corresponding to approx. date range: "
              f"{min(selected_dates_in_range).strftime('%Y-%m-%d')} to {max(selected_dates_in_range).strftime('%Y-%m-%d')}")
    else:
         print(f"Processing graphs from index {start_idx} to {end_idx}. No valid dates found within this specific index range.")


    return start_idx, end_idx, actual_max_dates

def preprocess_communities(partitions: List, extended_graphs: List, label_of_interest: str,
                           tracking_label: str, recording_label: str, weight_attr: str) -> List[List[CommunityData]]:
    """Preprocesses partitions and graphs into a list of CommunityData lists."""
    all_communities = []
    num_graphs = len(extended_graphs)

    for t, (partition, graph) in enumerate(zip(partitions, extended_graphs)):
        timepoint_communities = []
        if not partition or not graph or not hasattr(graph, 'vs'):
            all_communities.append(timepoint_communities) # Append empty list for this timepoint
            continue

        # Process communities for this timepoint


        for cid, community_members_indices in enumerate(partition): # Assuming partition gives indices or names
            comm_data = CommunityData(id=cid)
            node_mutations = set()
            current_max_weight = -np.inf # Use -inf for proper comparison
            core_node_label = "Unknown"
            core_node_muts = ""

            # Convert member indices to string names
            try:
                member_names = {graph.vs[mem_idx]['name'] for mem_idx in community_members_indices if 'name' in graph.vs[mem_idx].attributes()}
            except (IndexError, KeyError, TypeError, AttributeError):
                member_names = {str(m) for m in community_members_indices}

            comm_data.members = member_names

            for member_id_str in comm_data.members:
                try:
                    node = graph.vs.find(name=member_id_str)

                    # Collect mutations (assuming 'name' holds the mutation string)
                    mutations = node.attributes().get('name')
                    if mutations and isinstance(mutations, str):
                        node_mutations.add(mutations.strip())

                    # Check for target label
                    if not comm_data.has_target:
                        label_val = node.attributes().get(tracking_label)
                        if label_val is not None:
                            if isinstance(label_val, list):
                                if label_of_interest in label_val:
                                    comm_data.has_target = True
                            elif label_val == label_of_interest:
                                comm_data.has_target = True

                    # Record lineage counts
                    lineage = node.attributes().get(recording_label)
                    if lineage:
                        comm_data.lineage_counts[lineage] = comm_data.lineage_counts.get(lineage, 0) + 1

                    # Find core node based on weight
                    weight = node.attributes().get(weight_attr)
                    if weight is not None:
                        try:
                            weight_float = float(weight)
                            if weight_float > current_max_weight:
                                current_max_weight = weight_float
                                core_node_label = lineage if lineage else "Unknown"
                                core_node_muts = mutations if mutations else ""
                        except (ValueError, TypeError):
                            pass

                except (ValueError, KeyError, AttributeError):
                    pass

            comm_data.mutations = node_mutations
            comm_data.core_node = core_node_label
            comm_data.core_mutations = core_node_muts
            timepoint_communities.append(comm_data)

        all_communities.append(timepoint_communities)

    return all_communities

# ================ Tracking Engine ================
class CommunityTrackingEngine:
    """Optimized engine for tracking community evolution."""
    def __init__(self, config: TrackingConfig, similarity_calculator: SimilarityCalculator):
        self.config = config
        self.calculator = similarity_calculator
        self.time_window = config.time_window
        self.similarity_threshold = config.similarity_threshold
        self.size_filter_threshold = config.size_filter_threshold

    def track(self, preprocessed_communities: List[List[CommunityData]]) -> Tuple[Set[Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]], Dict[Tuple[int, int], List[Tuple[int, int]]], Dict[Tuple[int, int, int, int], float]]:
        """Tracks communities backward from target labels."""
        start_time = time.time()

        if not preprocessed_communities:
             print("Warning: No preprocessed communities to track.")
             return set(), {}, {}, {}

        community_sizes = self._calculate_community_sizes(preprocessed_communities)
        target_communities = self._identify_target_communities(preprocessed_communities)

        if not target_communities:
            print("Warning: No target communities found to initiate tracking.")
            return set(), {}, {}, {}

        chain_communities = set(target_communities)
        predecessors = {}
        alternative_predecessors = {}
        similarity_values = {}

        processed = set()
        to_process = list(target_communities)

        print(f"Starting tracking from {len(target_communities)} target communities...")

        iteration = 0
        while to_process:
            iteration += 1
            batch_current = [node for node in to_process if node not in processed]
            if not batch_current:
                break

            processed.update(batch_current)
            to_process = [node for node in to_process if node not in processed]

            community_pairs = self._filter_community_pairs(batch_current, preprocessed_communities, community_sizes)
            if not community_pairs:
                continue

            all_results = self._calculate_similarities(community_pairs, preprocessed_communities)

            new_to_process = self._process_matching_results(
                all_results, predecessors, alternative_predecessors,
                similarity_values, chain_communities, processed
            )

            to_process.extend(new_to_process)


        self._report_tracking_results(chain_communities, target_communities, start_time)
        return chain_communities, predecessors, alternative_predecessors, similarity_values

    def _calculate_community_sizes(self, communities: List[List[CommunityData]]) -> Dict[Tuple[int, int], int]:
        """Calculates and stores the size (number of members) of each community."""
        return {
            (t_idx, c_idx): len(comm.members)
            for t_idx, communities_t in enumerate(communities)
            for c_idx, comm in enumerate(communities_t)
        }

    def _identify_target_communities(self, communities: List[List[CommunityData]]) -> Set[Tuple[int, int]]:
        """Identifies communities containing the target label."""
        return {
            (t_idx, c_idx)
            for t_idx, communities_t in enumerate(communities)
            for c_idx, comm in enumerate(communities_t)
            if comm.has_target
        }

    def _filter_community_pairs(self, batch_current: List[Tuple[int, int]], preprocessed_communities: List[List[CommunityData]], community_sizes: Dict[Tuple[int, int], int]) -> List[Tuple[int, int, int, int]]:
        """Filters potential predecessor pairs based on time window and size ratio."""
        candidate_pairs = []
        max_t_idx = len(preprocessed_communities) - 1

        for t_idx, c_idx in batch_current:
            current_size = community_sizes.get((t_idx, c_idx), 0)
            if current_size == 0:
                continue

            min_prev_t_idx = max(0, t_idx - self.time_window)
            for prev_t_idx in range(min_prev_t_idx, t_idx):
                if prev_t_idx >= len(preprocessed_communities): 
                    continue

                for prev_c_idx, prev_comm in enumerate(preprocessed_communities[prev_t_idx]):
                    prev_size = community_sizes.get((prev_t_idx, prev_c_idx), 0)

                    if prev_size == 0: 
                        continue
                    size_ratio = min(current_size, prev_size) / max(current_size, prev_size)
                    if size_ratio < self.size_filter_threshold:
                        continue

                    candidate_pairs.append((t_idx, c_idx, prev_t_idx, prev_c_idx))

        return candidate_pairs

    def _calculate_similarities(self, community_pairs: List[Tuple[int, int, int, int]], preprocessed_communities: List[List[CommunityData]]) -> List[Tuple[int, int, int, int, float]]:
        """Calculates similarities for a list of community pairs, potentially in batches."""
        if not community_pairs:
            return []

        batch_args = ((preprocessed_communities, self.similarity_threshold), community_pairs)
        results = self.calculator.batch_calculate(batch_args)

        return results # List of (t, c, prev_t, prev_c, sim) where sim > threshold


    def _process_matching_results(self, all_results: List[Tuple[int, int, int, int, float]],
                                predecessors: Dict[Tuple[int, int], Tuple[int, int]],
                                alternative_predecessors: Dict[Tuple[int, int], List[Tuple[int, int]]],
                                similarity_values: Dict[Tuple[int, int, int, int], float],
                                chain_communities: Set[Tuple[int, int]],
                                processed: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Processes similarity results to find best predecessors and identify new communities to trace."""
        # Group results by the current community (t_idx, c_idx)
        matches_by_current: Dict[Tuple[int, int], List[Tuple[int, int, float, int]]] = {}
        for t_idx, c_idx, prev_t_idx, prev_c_idx, sim in all_results:
            current_key = (t_idx, c_idx)
            if current_key not in matches_by_current:
                matches_by_current[current_key] = []
            time_diff = t_idx - prev_t_idx # Store time difference for sorting
            matches_by_current[current_key].append((prev_t_idx, prev_c_idx, sim, time_diff))
            similarity_values[(t_idx, c_idx, prev_t_idx, prev_c_idx)] = sim

        new_to_process = []
        processed_in_this_step = set() # Track newly assigned predecessors

        for current_key, matches in matches_by_current.items():
            if not matches or current_key in predecessors: # Skip if already has a predecessor
                continue

            # Sort matches: primary key = time difference (ascending), secondary key = similarity (descending)
            matches.sort(key=lambda x: (x[3], -x[2]))

            # The best match is the first one after sorting
            best_match = matches[0]
            best_prev_t_idx, best_prev_c_idx, best_sim, best_time_diff = best_match
            best_prev_key = (best_prev_t_idx, best_prev_c_idx)

            # Assign the best predecessor
            predecessors[current_key] = best_prev_key
            processed_in_this_step.add(current_key)

            # Find alternatives (same time difference, very close similarity)
            alternatives = []
            for prev_t, prev_c, sim, time_diff in matches[1:]:
                if time_diff == best_time_diff and abs(sim - best_sim) < 1e-6:
                     alternatives.append((prev_t, prev_c))
                elif time_diff > best_time_diff: # Stop checking once time difference increases
                    break
            if alternatives:
                alternative_predecessors[current_key] = alternatives

            # Add the predecessor to the chain and potentially to the processing queue
            chain_communities.add(best_prev_key)
            if best_prev_key not in processed and best_prev_key not in predecessors: # Only add if not processed and not already assigned its own predecessor in this batch

                new_to_process.append(best_prev_key)


        # Return unique new nodes to process
        return list(set(new_to_process))


    def _report_tracking_results(self, chain_communities: Set[Tuple[int, int]], target_communities: Set[Tuple[int, int]], start_time: float):
        """Prints a summary of the tracking results."""
        tracked_count = len(chain_communities) - len(target_communities)
        print(f"Identified {len(chain_communities)} communities in the evolution chain "
              f"({len(target_communities)} targets + {max(0, tracked_count)} tracked predecessors). "
              f"Tracking took {time.time() - start_time:.2f}s.")

# ================ Tracking Result Processing ================
class TrackingResultProcessor:
    """Processes raw tracking results into structured data."""
    def __init__(self, config: TrackingConfig, date_range: List[pd.Timestamp]):
        self.config = config
        self.date_range = date_range # List of actual dates corresponding to time indices

    def process_timepoints(self, t_rel: int, actual_date: pd.Timestamp,
                           preprocessed_communities: List[List[CommunityData]],
                           chain_communities: Set[Tuple[int, int]],
                           predecessors: Dict[Tuple[int, int], Tuple[int, int]],
                           alternative_predecessors: Dict[Tuple[int, int], List[Tuple[int, int]]],
                           similarity_values: Dict[Tuple[int, int, int, int], float],
                           start_idx: int = 0) -> List[TrackingNode]: # start_idx is no longer used here for lookups
        """Processes a single timepoint to create TrackingNode objects using relative indices."""
        # t_rel is the relative index within the processed slice (0 to len(actual_dates)-1)
        # t is the original absolute index if needed (start_idx + t_rel) - calculated if needed for output, not lookup
        tracking_nodes = []

        # Use t_rel for accessing the slice
        if t_rel >= len(self.date_range) or t_rel >= len(preprocessed_communities) or not preprocessed_communities[t_rel]:
            return tracking_nodes # Out of bounds or no communities at this timepoint

        current_communities_at_t_rel = preprocessed_communities[t_rel]

        # Find which communities at this relative timepoint 't_rel' are part of the final chains
        # chain_communities should contain relative indices (t_rel, c_idx) from the engine
        chain_indices_at_t_rel = {c_idx for t_idx_rel, c_idx in chain_communities if t_idx_rel == t_rel}

        if not chain_indices_at_t_rel:
            return tracking_nodes # No communities from this timepoint are in the chains

        for c_idx in chain_indices_at_t_rel:
             if c_idx >= len(current_communities_at_t_rel): continue # Index safety check

             comm_data = current_communities_at_t_rel[c_idx]
             current_key_rel = (t_rel, c_idx) # Use relative index key

             # Look up predecessors using the relative key
             best_match_rel = predecessors.get(current_key_rel) # Should contain relative (prev_t_rel, prev_c_idx)
             alternative_matches_rel = alternative_predecessors.get(current_key_rel, []) # Should be list of relative indices

             similarity = 0.0
             if best_match_rel:
                 prev_t_idx_rel, prev_c_idx_abs = best_match_rel # prev_c_idx_abs is likely just prev_c_idx
                 # Look up similarity using relative keys
                 similarity = similarity_values.get((t_rel, c_idx, prev_t_idx_rel, prev_c_idx_abs), 0.0)

             # Store relative indices in TrackingNode
             tracking_nodes.append(
                 TrackingNode(
                     date=actual_date, # Use the actual date for this timepoint
                     community_id=comm_data.id, # Use the original community ID
                     lineage_counts=comm_data.lineage_counts,
                     similarity=similarity,
                     matched_from=best_match_rel, # Store relative (t_rel, c_idx)
                     alternative_matched_from=alternative_matches_rel, # Store relative list
                     contains_label=comm_data.has_target,
                     core_node=comm_data.core_node,
                     core_node_mutations=comm_data.core_mutations
                 )
             )

        return tracking_nodes

    def build_tracking_dataframe(self, tracking_nodes_dict: Dict[int, List[TrackingNode]], recording_label: str = "Lineage", start_idx: int = 0) -> pd.DataFrame: # Pass start_idx here
        """Builds a DataFrame from the processed TrackingNode objects."""
        rows = []
        # Build date map using absolute index as key
        abs_date_map = {start_idx + t_rel: date.strftime('%Y-%m-%d')
                        for t_rel, date in enumerate(self.date_range)}


        # Iterate through relative time index (t_rel) and list of nodes
        for t_rel, nodes_at_t_rel in tracking_nodes_dict.items():
            current_abs_t = start_idx + t_rel # Calculate absolute time index for current node
            for nd in nodes_at_t_rel:
                pcid = pd.NA
                pt_abs = pd.NA # Absolute time index of predecessor
                matched_from_date_str = pd.NA

                if nd.matched_from:
                    # nd.matched_from contains relative (pt_rel, pc_idx)
                    pt_rel, pc_idx = nd.matched_from
                    pcid = pc_idx
                    pt_abs = start_idx + pt_rel # Calculate absolute time index for predecessor
                    matched_from_date_str = abs_date_map.get(pt_abs) # Get date string using absolute index


                # Alternative matched from processing
                alt_matched_list_abs = []
                # Assume alternative_matched_from stores relative (alt_t_rel, alt_c)
                for alt_t_rel, alt_c in getattr(nd, 'alternative_matched_from', []):
                     alt_t_abs = start_idx + alt_t_rel # Calculate absolute index
                     alt_matched_list_abs.append((alt_t_abs, alt_c)) # Store as (abs_t, abs_c)

                rows.append({
                    "Date": nd.date.strftime('%Y-%m-%d'),
                    "Community ID": nd.community_id,
                    "Contains Label": nd.contains_label,
                    "Similarity": f"{nd.similarity:.3f}" if pd.notna(nd.similarity) else pd.NA,
                    f"{recording_label} Counts": json.dumps(nd.lineage_counts, sort_keys=True) if nd.lineage_counts else json.dumps({}),
                    "Matched From Time Index": pt_abs, # Store Absolute Time Index
                    "Matched From Date": matched_from_date_str, # Store Date String
                    "Matched From Community ID": pcid,
                    "Core Node": nd.core_node,
                    "Mutations_of_core_node": nd.core_node_mutations,
                    # Store alternatives as list of tuples (abs_t, abs_c)
                    "Alternative Matched From": json.dumps(alt_matched_list_abs)
                })

        df = pd.DataFrame(rows)
        # ... (rest of the type conversion and JSON parsing remains the same) ...
        # Remove the explicit conversion for 'Matched From Date' as it's added directly now
        if "Matched From Time Index" in df.columns:
             df["Matched From Time Index"] = pd.to_numeric(df["Matched From Time Index"], errors='coerce').astype('Int64')


        return df

def process_tracking_results(preprocessed_communities: List[List[CommunityData]],
                             chain_communities: Set[Tuple[int, int]],
                             predecessors: Dict[Tuple[int, int], Tuple[int, int]],
                             alternative_predecessors: Dict[Tuple[int, int], List[Tuple[int, int]]],
                             similarity_values: Dict[Tuple[int, int, int, int], float],
                             actual_dates: List[pd.Timestamp], # Dates corresponding to preprocessed_communities slice
                             result_processor: TrackingResultProcessor,
                             recording_label: str,
                             start_idx: int = 0) -> pd.DataFrame: # Pass global start_idx
    """Processes tracking results into a DataFrame using TrackingResultProcessor."""
    tracking_nodes_dict = {} # Keyed by relative time index (t_rel)

    # Iterate through the timepoints of the *processed slice*
    for t_rel, actual_date in enumerate(actual_dates):
        # Pass t_rel and actual_date to process_timepoints
        nodes = result_processor.process_timepoints(
            t_rel, actual_date,
            preprocessed_communities, chain_communities,
            predecessors, alternative_predecessors, similarity_values
        )
        if nodes: # Only add if non-empty
             tracking_nodes_dict[t_rel] = nodes

    # Pass start_idx to build_tracking_dataframe for correct absolute index calculation
    df_track = result_processor.build_tracking_dataframe(tracking_nodes_dict, recording_label, start_idx)

    return df_track

# ================ Chain Building ================
def kmp_build_table(pattern: tuple) -> List[int]:
    """Builds the partial match table for KMP algorithm."""
    m = len(pattern)
    table = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            table[i] = length
            i += 1
        else:
            if length != 0:
                length = table[length - 1]
            else:
                table[i] = 0
                i += 1
    return table

def is_continuous_subsequence_kmp(shorter: tuple, longer: tuple) -> bool:
    """Checks if 'shorter' is a continuous subsequence of 'longer' using KMP."""
    if not shorter: return True
    m, n = len(shorter), len(longer)
    if m > n: return False
    if m == 0: return True
    if m == n: return shorter == longer

    # KMP for longer patterns
    lps = kmp_build_table(shorter)
    i = j = 0
    while i < n:
        if shorter[j] == longer[i]:
            i += 1
            j += 1
        if j == m:
            return True # Found pattern
        elif i < n and shorter[j] != longer[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return False

def build_chain(start_node_key: Tuple[str, int], # (DateString, CommunityID)
                node_records: Dict[Tuple[str, int], pd.Series], # Map key to row data
                successors_map: Dict[Tuple[str, int], List[Tuple[str, int, float]]], # Key -> List of (succ_date, succ_cid, sim)
                predecessors_map: Dict[Tuple[str, int], Tuple[str, int]], # Key -> (pred_date, pred_cid)
                recording_label: str # Add recording_label argument
               ) -> Optional[pd.DataFrame]:
    """Builds a single forward and backward chain starting from a node."""
    chain_rows = []
    visited_forward = set()
    visited_backward = set() # Keep track of nodes added during backward traversal

    # --- Build Forward ---
    current_key_fwd = start_node_key
    forward_part = []
    while current_key_fwd and current_key_fwd in node_records and current_key_fwd not in visited_forward:
        visited_forward.add(current_key_fwd)
        node_data = node_records[current_key_fwd]
        forward_part.append(node_data)

        successors = successors_map.get(current_key_fwd, [])
        if not successors:
            break # End of chain

        successors.sort(key=lambda x: x[2], reverse=True) # Sort by similarity descending
        next_date_str, next_cid, _ = successors[0]
        current_key_fwd = (next_date_str, next_cid)

    # --- Build Backward ---
    # Start traversal *from* the start_node_key's predecessor
    current_key_bwd = start_node_key
    backward_part = []
    # The start_node_key itself is handled by the forward pass or added later if needed.
    # visited_backward tracks nodes *already added* to backward_part to prevent cycles.

    while True: # Loop until break condition is met
         pred_key = predecessors_map.get(current_key_bwd)

         # Break conditions: no predecessor, predecessor data missing, or predecessor already added to backward chain
         if not pred_key or pred_key not in node_records or pred_key in visited_backward:
              break

         # Check if the predecessor was part of the forward chain (merge point)
         if pred_key in visited_forward:
             # print(f"Backward chain merged with forward chain at {pred_key}") # Optional debug
             break # Stop backward traversal here

         visited_backward.add(pred_key) # Mark this predecessor as added to the backward chain
         node_data = node_records[pred_key]
         backward_part.append(node_data)
         current_key_bwd = pred_key # Move to the next predecessor

    # Combine backward (reversed) and forward parts
    # The forward_part already contains the start_node_key if it was reachable
    chain_rows = backward_part[::-1] + forward_part

    if not chain_rows:
        if start_node_key in node_records and not any(r.name == start_node_key for r in forward_part): # Check if start node is missing
             # This logic might be complex depending on desired behavior for isolated nodes.
             # For now, assume if chain_rows is empty, return None.
             return None


    df_chain = pd.DataFrame(chain_rows)

    # --- The rest of the function remains the same ---
    try:
        # Dynamically determine the lineage counts column name
        lineage_col_name = f"{recording_label} Counts"

        # Check if the dynamically generated column name exists
        if lineage_col_name not in df_chain.columns:
             print(f"Warning: Column '{lineage_col_name}' not found in chain starting at {start_node_key}. Columns: {df_chain.columns}. Skipping.")
             return None # Fail if the specific column is missing

        # Define desired column order (Date will become index)
        desired_columns_in_order = [
            'Community ID',
            'Contains Label',
            'Similarity',
            lineage_col_name, # Use dynamic name
            'Matched From Date',
            'Matched From Community ID',
            'Core Node',
            'Mutations_of_core_node',
            'Matched From Time Index',
            'Alternative Matched From'
        ]

        # Ensure 'Date' column exists before attempting to use it
        if 'Date' not in df_chain.columns:
             print(f"Warning: 'Date' column missing in chain starting at {start_node_key}. Cannot set index.")
             return None # Cannot proceed without Date column for indexing

        # Select existing columns in the desired order, keeping 'Date' temporarily
        final_columns = [col for col in desired_columns_in_order if col in df_chain.columns]
        # Ensure unique rows if start node was potentially added twice (unlikely with current logic but safer)
        # df_chain = df_chain.loc[~df_chain.index.duplicated(keep='first')] # Requires index to be set first

        df_final = df_chain[['Date'] + final_columns].copy()
        # Drop duplicates based on Date and Community ID before setting index
        df_final.drop_duplicates(subset=['Date', 'Community ID'], keep='first', inplace=True)


        df_final.set_index('Date', inplace=True)

        # Sort by index (Date strings - lexicographical sort)
        df_final.sort_index(inplace=True)

        # Final check for label presence (using the correct column name)
        if 'Contains Label' not in df_final.columns or not df_final['Contains Label'].any():
            # print(f"Chain starting at {start_node_key} discarded: No 'Contains Label' == True.") # Optional Debug
            return None

        return df_final

    except KeyError as e:
        print(f"Warning: Missing expected column '{e}' during chain finalization for {start_node_key}. Columns: {df_chain.columns}. Skipping chain.")
        return None
    except Exception as e:
         print(f"Warning: Error processing chain DataFrame for {start_node_key}: {e}. Skipping chain.")
         return None
    
def build_chain_worker(args):
    """Wrapper for parallel execution of build_chain."""
    start_node_key, node_records, successors_map, predecessors_map, recording_label = args
    try:
        return build_chain(start_node_key, node_records, successors_map, predecessors_map, recording_label)
    except Exception as e:
        return None

class ChainBuilder:
    """Builds and deduplicates community evolution chains."""
    def __init__(self, config: TrackingConfig, date_range: List[pd.Timestamp], recording_label: str): # Add recording_label
        self.config = config
        self.date_range = date_range
        self.n_processes = config.n_processes
        self.recording_label = recording_label

    def build_chains(self, df_track: pd.DataFrame,
                    ) -> List[pd.DataFrame]:
        """Builds and deduplicates chains from the tracking DataFrame."""
        if df_track.empty:
            print("Tracking DataFrame is empty, cannot build chains.")
            return []

        chain_start = time.time()

        node_records, successors_map, predecessors_map = self._build_tracking_indices(df_track)

        root_nodes, labeled_nodes = self._identify_special_nodes(node_records, predecessors_map)

        start_nodes = set(root_nodes) | set(labeled_nodes)
        print(f"Identified {len(root_nodes)} root nodes and {len(labeled_nodes)} labeled nodes. "
              f"Total {len(start_nodes)} unique start points for chain building.")

        if not start_nodes:
             print("No start nodes found for chain building.")
             return []

        tasks = [(node_key, node_records, successors_map, predecessors_map, self.recording_label) for node_key in start_nodes]

        all_chains_raw = []
        if len(tasks) > self.n_processes * 2 and self.n_processes > 1:
            print(f"Building chains in parallel using {self.n_processes} processes...")
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                chunksize = max(1, len(tasks) // (self.n_processes * 4))
                all_chains_raw = list(executor.map(build_chain_worker, tasks, chunksize=chunksize))
        else:
            print("Building chains sequentially...")
            all_chains_raw = [build_chain_worker(task) for task in tasks]

        all_chains_filtered = [chain for chain in all_chains_raw if chain is not None and not chain.empty]

        if not all_chains_filtered:
            print("No valid chains containing the label were constructed after build_chain.")
            return []

        print(f"Constructed {len(all_chains_filtered)} raw chains containing the label.")

        unique_chains = self._deduplicate_chains(all_chains_filtered)

        print(f"Reduced to {len(unique_chains)} unique chains after deduplication. "
              f"Chain construction took {time.time() - chain_start:.2f}s.")

        return unique_chains

    def _build_tracking_indices(self, df_track: pd.DataFrame) -> Tuple[Dict[Tuple[str, int], pd.Series], Dict[Tuple[str, int], List[Tuple[str, int, float]]], Dict[Tuple[str, int], Tuple[str, int]]]:
        """Builds dictionaries for fast lookup using (DateString, CommunityID) keys."""
        node_records = {}
        successors_map = {}
        predecessors_map = {}

        for _, row in df_track.iterrows():
            try:
                cid = int(row['Community ID'])
                date_str = row['Date']
                key = (date_str, cid)
                node_records[key] = row

                if pd.notna(row['Matched From Date']) and pd.notna(row['Matched From Community ID']):
                    pred_date_str = row['Matched From Date']
                    pred_cid = int(row['Matched From Community ID'])
                    pred_key = (pred_date_str, pred_cid)
                    predecessors_map[key] = pred_key

                    sim = float(row['Similarity']) if pd.notna(row['Similarity']) else 0.0
                    if pred_key not in successors_map:
                        successors_map[pred_key] = []
                    successors_map[pred_key].append((date_str, cid, sim))

            except (ValueError, TypeError, KeyError):
                pass

        return node_records, successors_map, predecessors_map

    def _identify_special_nodes(self, node_records: Dict[Tuple[str, int], pd.Series], predecessors_map: Dict[Tuple[str, int], Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Identifies root nodes and labeled nodes."""
        root_nodes = []
        labeled_nodes = []

        for key, row in node_records.items():
            is_root = True
            if 'Matched From Date' in row and pd.notna(row['Matched From Date']):
                if 'Matched From Community ID' in row and pd.notna(row['Matched From Community ID']):
                    is_root = False

            if is_root:
                root_nodes.append(key)

            if row.get('Contains Label', False):
                labeled_nodes.append(key)

        return root_nodes, labeled_nodes

    def _deduplicate_chains(self, chains: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Removes chains that are subsequences of longer chains using KMP."""
        if len(chains) <= 1:
            return chains

        # Represent chains as tuples of keys (Index=Date String, CommunityID) for efficient comparison
        chain_tuples = []
        valid_chains_for_dedup = []
        original_indices = [] # Keep track of original index in 'chains' list

        for i, chain in enumerate(chains):
            # Check if chain is valid and has an index (no longer checking for DatetimeIndex)
            if chain is None or chain.empty or chain.index is None: # MODIFIED CHECK
                # print(f"Skipping chain {i} in deduplication due to invalid format.")
                continue
            try:
                # Ensure Community ID is treated as integer for the tuple key
                current_tuple = tuple(zip(chain.index, chain['Community ID'].astype(int))) # MODIFIED TUPLE CREATION
                chain_tuples.append(current_tuple)
                valid_chains_for_dedup.append(chain)
                original_indices.append(i) # Store original index
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                print(f"Error converting chain {i} to tuple for deduplication: {e}. Skipping this chain.")
                continue # Skip chains that cause errors during conversion

        if len(valid_chains_for_dedup) <= 1:
            return valid_chains_for_dedup # Return the valid chains if only 0 or 1 left

        # Sort by length descending, then by start date string ascending as tie-breaker
        indexed_chain_tuples = list(enumerate(chain_tuples)) # Use index within the valid list
        # Sort by start date string (first element of the first tuple)
        indexed_chain_tuples.sort(key=lambda x: x[1][0][0] if x[1] else '')
        # Primary sort: length descending
        indexed_chain_tuples.sort(key=lambda x: len(x[1]), reverse=True)


        n_chains = len(indexed_chain_tuples)
        is_subset = [False] * n_chains # Tracks subset status within the valid_chains_for_dedup list
        final_indices_in_valid_list = [] # Stores indices *within the valid list*

        for i in range(n_chains):
            # Get the index in the valid list for the current chain being considered
            current_valid_list_idx = indexed_chain_tuples[i][0]
            if is_subset[current_valid_list_idx]: # Check if this chain was marked as subset
                continue

            final_indices_in_valid_list.append(current_valid_list_idx) # Keep this chain
            chain_i_tuple = indexed_chain_tuples[i][1]

            # Check subsequent chains (which are shorter or same length)
            for j in range(i + 1, n_chains):
                # Get the index in the valid list for the chain being compared
                compare_valid_list_idx = indexed_chain_tuples[j][0]
                if is_subset[compare_valid_list_idx]: # Skip if already marked as subset
                    continue

                chain_j_tuple = indexed_chain_tuples[j][1]

                # Use KMP to check if chain_j is a subsequence of chain_i
                if is_continuous_subsequence_kmp(chain_j_tuple, chain_i_tuple):
                    # Mark chain j (at index compare_valid_list_idx in the valid list) as a subset
                    is_subset[compare_valid_list_idx] = True


        # Retrieve the original DataFrame objects using the original indices mapping
        unique_chains_list = []
        for valid_list_idx in final_indices_in_valid_list:
            # Map the index from the valid list back to the index in the original 'chains' list
            original_chain_idx = original_indices[valid_list_idx]
            unique_chains_list.append(chains[original_chain_idx]) # Append the actual DataFrame


        return unique_chains_list

# ================ Main Entry Point ================
def track_community_evolution(
    partitions: List, # List of community partitions (e.g., list of lists of node indices/names)
    extended_graphs: List, # List of graph objects (e.g., igraph Graphs)
    label_of_interest: str, # Target label value to track
    tracking_label: str, # Node attribute name holding the label to check against label_of_interest
    recording_label: str, # Node attribute name holding the lineage/type to record counts for
    start_date: Optional[str] = None, # Start date string 'YYYY-MM-DD' (optional)
    end_date: Optional[str] = None, # End date string 'YYYY-MM-DD' (optional)
    time_interval: int = 7, # Assumed interval between graphs if dates are missing (days)
    similarity_threshold: float = 0.4, # Threshold for matching communities
    weight_attr: str = "wdks", # Node attribute for determining core node
    n_processes: int = 4, # Number of processes for parallel tasks (chain building)
    time_window: int = 2, # Lookback window (in time steps) for finding predecessors
    use_gpu: bool = True, # Whether to attempt using GPU for similarity calculation
    output_path: Optional[str] = None # Path to save results (.h5 file). If None, uses default.
) -> List[pd.DataFrame]:
    """
    Tracks community evolution based on similarity and a target label.

    Args:
        partitions: List of community partitions for each time step.
        extended_graphs: List of corresponding graph objects with node attributes.
        label_of_interest: The specific value in the `tracking_label` attribute to track.
        tracking_label: The node attribute key used to find the `label_of_interest`.
        recording_label: The node attribute key used for counting lineages/types within communities.
        start_date: Optional start date ('YYYY-MM-DD') to filter data.
        end_date: Optional end date ('YYYY-MM-DD') to filter data.
        time_interval: Assumed days between graph snapshots if dates are missing.
        similarity_threshold: Minimum similarity score to consider communities matched.
        weight_attr: Node attribute used to identify the 'core' node of a community.
        n_processes: Number of CPU cores for parallel processing steps.
        time_window: How many previous time steps to search for potential predecessors.
        use_gpu: If True, attempts to use GPU for similarity calculations.
        output_path: Optional path to save the resulting list of chain DataFrames in HDF5 format.

    Returns:
        A list of pandas DataFrames, where each DataFrame represents a unique evolution chain
        containing the label of interest. Returns an empty list if no chains are found or an error occurs.
    """
    overall_start = time.time()

    config = TrackingConfig(
        similarity_threshold=similarity_threshold,
        time_window=time_window,
        use_gpu=use_gpu,
        n_processes=max(1, n_processes),
        weight_attr=weight_attr,
    )

    start_idx, end_idx, actual_dates = extract_date_range(
        extended_graphs, start_date, end_date, time_interval)

    if start_idx == -1 or actual_dates is None:
        print("Error: Could not determine a valid processing range based on dates and graph data. Aborting.")
        return []
    if not actual_dates:
         print("Warning: No actual dates could be determined for the selected graph range.")


    partitions_slice = partitions[start_idx : end_idx + 1]
    graphs_slice = extended_graphs[start_idx : end_idx + 1]
    if len(actual_dates) != len(graphs_slice):
        print(f"Warning: Mismatch between number of actual dates ({len(actual_dates)}) and selected graphs ({len(graphs_slice)}). Using graph count.")

    preprocessed = preprocess_communities(
        partitions_slice, graphs_slice,
        label_of_interest, tracking_label, recording_label, weight_attr)


    if not any(preprocessed):
        print("Warning: Preprocessing resulted in no community data. Aborting.")
        return []

    calculator = create_calculator(config)
    engine = CommunityTrackingEngine(config, calculator)

    track_results = engine.track(preprocessed)
    chain_communities, predecessors, alternative_predecessors, similarity_values = track_results

    if not chain_communities:
        print("Tracking finished, but no communities were linked into chains. No results to process.")
        return []

    result_processor = TrackingResultProcessor(config, actual_dates)
    df_track = process_tracking_results(
        preprocessed, chain_communities, predecessors,
        alternative_predecessors, similarity_values,
        actual_dates, result_processor, recording_label, start_idx=0
    )

    if df_track.empty:
        print("Warning: Tracking data was generated, but processing resulted in an empty DataFrame.")
        return []
    print(f"Generated tracking DataFrame with {len(df_track)} rows.")


    chain_builder = ChainBuilder(config, actual_dates, recording_label) 
    final_chains = chain_builder.build_chains(df_track)

    if not final_chains:
        print("No final chains containing the label were constructed.")
        return []

    final_chains.sort(key=lambda chain: not (chain['Contains Label'] == True).any(), reverse=False)

    if output_path is None:
        date_range_str = ""
        if actual_dates:
            start_str = actual_dates[0].strftime('%Y%m%d')
            end_str = actual_dates[-1].strftime('%Y%m%d')
            date_range_str = f"_{start_str}_{end_str}"
        elif start_date and end_date:
            date_range_str = f"_{start_date.replace('-', '')}_{end_date.replace('-', '')}"

        output_path = f"./tracking_results_label_{label_of_interest}{date_range_str}.h5"

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        compatible_chains = []
        for i, df in enumerate(final_chains):
            if df is not None and not df.empty:
                df_copy = df.copy()
                for col in df_copy.select_dtypes(include=['object']).columns:
                    try:
                        df_copy[col] = df_copy[col].astype(str)
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col}' in chain {i} to string: {e}")
                compatible_chains.append(df_copy)

        IOManager.save_to_hdf5(compatible_chains, output_path, num_threads=n_processes)
        print(f"Tracking results successfully saved to {output_path}.")

    except ImportError:
         print("Error: Could not import IOManager from TempSnap. Results not saved.")
    except Exception as e:
        print(f"Error saving tracking results to {output_path}: {e}")


    print(f"Total processing time: {time.time() - overall_start:.2f}s")
    return final_chains

# ================ Community Node Extraction ================
class CommunityNodeExtractor:
    """High performance extractor for community node data"""
    
    def __init__(self):
        """Initialize extractor with empty caches"""
        self.graph_cache = {}
        self.date_mapping_cache = {}
        self.processed_graph_ids = set()
    
    def _preprocess_graph(self, graph):
        """Transform graph into optimized lookup structure"""
        node_data = {}
        
        # Get all attributes at once
        all_attrs = {}
        for attr in graph.vs.attributes():
            all_attrs[attr] = graph.vs[attr]
        
        if 'name' not in all_attrs:
            return {}
        
        # Create node lookup table
        names = all_attrs['name']
        name_to_idx = {name: i for i, name in enumerate(names) if name}
        
        # Extract all node attributes except 'name' and 'kshell'
        attrs_to_keep = [a for a in graph.vs.attributes() if a != 'name' and a != 'kshell']
        
        for node_name, idx in name_to_idx.items():
            node_attrs = {attr: all_attrs[attr][idx] for attr in attrs_to_keep}
            index_key = node_attrs.get('Mutations_str', node_name)
            node_data[node_name] = (index_key, node_attrs)
        
        return node_data
    
    def _create_date_mapping(self, graphs, chains):
        """Create optimized date to graph index mapping"""
        # Implementation unchanged
        max_dates = []
        for i, g in enumerate(graphs):
            if not g or 'Date' not in g.vs.attributes():
                max_dates.append((i, None))
                continue
            
            date_values = [v['Date'] for v in g.vs if 'Date' in v.attributes() and v['Date']]
            if not date_values:
                max_dates.append((i, None))
                continue
            
            try:
                dates = pd.to_datetime(date_values, errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    max_dates.append((i, valid_dates.max()))
                else:
                    max_dates.append((i, None))
            except:
                max_dates.append((i, None))
        
        # Filter and sort by date
        valid_dates = [(i, d) for i, d in max_dates if d is not None]
        if not valid_dates:
            return {}
        
        valid_dates.sort(key=lambda x: x[1])
        date_indices = [i for i, _ in valid_dates]
        date_values = np.array([d.timestamp() for _, d in valid_dates])
        
        # Get dates from chains
        dates_to_map = set()
        for df in chains:
            if df.empty:
                continue
            try:
                valid_idx_dates = pd.to_datetime(df.index, errors='coerce').dropna()
                dates_to_map.update(d.strftime('%Y-%m-%d') for d in valid_idx_dates)
            except:
                pass
        
        # Create mapping with binary search
        mapping = {}
        for date_str in dates_to_map:
            try:
                target = pd.to_datetime(date_str).timestamp()
                idx = np.searchsorted(date_values, target, side='right') - 1
                if idx >= 0:
                    mapping[date_str] = date_indices[idx]
                elif len(date_indices) > 0:
                    mapping[date_str] = date_indices[0]
                else:
                    mapping[date_str] = 0
            except:
                mapping[date_str] = 0
        
        return mapping
    
    def extract_nodes(self, community_id, date_str, graphs, communities, chains, preload=False, debug=False):
        """Extract community node data with automatic caching"""
        start_time = time.time()
        
        # Handle preloading if requested
        if preload:
            graph_ids = set(id(g) for g in graphs if g is not None)
            new_graphs = graph_ids - self.processed_graph_ids
            
            if new_graphs:
                if debug:
                    print(f"Preloading {len(new_graphs)} new graphs...")
                
                for i, graph in enumerate(graphs):
                    if graph is None or id(graph) not in new_graphs:
                        continue
                    
                    cache_key = id(graph)
                    self.graph_cache[cache_key] = self._preprocess_graph(graph)
                    self.processed_graph_ids.add(cache_key)
        
        # Get date mapping (with caching)
        cache_key = (id(graphs), id(chains))
        if cache_key not in self.date_mapping_cache:
            self.date_mapping_cache[cache_key] = self._create_date_mapping(graphs, chains)
            
        date_mapping = self.date_mapping_cache[cache_key]
        
        if date_str not in date_mapping:
            if debug:
                print(f"Date {date_str} not found in mapping")
            return pd.DataFrame()
        
        graph_idx = date_mapping[date_str]
        
        # Validate indices
        if not (0 <= graph_idx < len(graphs) and graphs[graph_idx] and 
                0 <= graph_idx < len(communities) and 
                0 <= community_id < len(communities[graph_idx])):
            if debug:
                print(f"Invalid index: graph {graph_idx}, community {community_id}")
            return pd.DataFrame()
        
        graph = graphs[graph_idx]
        community_nodes = communities[graph_idx][community_id]
        
        if not community_nodes:
            if debug:
                print(f"Community {community_id} is empty")
            return pd.DataFrame()
        
        # Get or create preprocessed graph data
        graph_cache_key = id(graph)
        if graph_cache_key not in self.graph_cache:
            self.graph_cache[graph_cache_key] = self._preprocess_graph(graph)
            self.processed_graph_ids.add(graph_cache_key)
            
        node_data_map = self.graph_cache[graph_cache_key]
        
        # Extract node attributes
        index_keys = []
        node_attrs_list = []
        
        for node_name in community_nodes:
            if node_name in node_data_map:
                index_key, attrs = node_data_map[node_name]
                index_keys.append(index_key)
                node_attrs_list.append(attrs)
        
        # Build DataFrame
        if not node_attrs_list:
            if debug:
                print("No valid nodes found")
            return pd.DataFrame()
        
        df = pd.DataFrame(node_attrs_list, index=index_keys)
        df.index.name = 'Mutations_str'
        
        # Set column order - ensure Ancestor is included, kshell is excluded
        priority_cols = ['Date', 'ID', 'Clade', 'Lineage', 'Location', 'Ancestor', 'Ancestor_ID']
        present_priority = [col for col in priority_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in priority_cols and col != 'kshell']
        
        # Ensure wdks column is included in results if present
        if 'wdks' in df.columns and 'wdks' not in other_cols and 'wdks' not in present_priority:
            other_cols.append('wdks')
            
        df = df[present_priority + sorted(other_cols)]
        
        if debug:
            print(f"Extracted {len(df)} nodes in {time.time() - start_time:.4f}s")
        
        return df

# Create singleton instance for convenient access
_extractor = CommunityNodeExtractor()

def extract_community_nodes(community_id, date_str, graphs, communities, chains, preload=False, debug=False):
    """Extract node data for a community with simplified interface"""
    return _extractor.extract_nodes(
        community_id, date_str, graphs, communities, chains, preload, debug)