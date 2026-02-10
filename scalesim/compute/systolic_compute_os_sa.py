"""
This module implements the 'systolic_compute_is' class, which simulates a systolic array with Output
Stationary (OS) dataflow. It handles operand prefetching, demand matrix creation, and performance
metrics such as mapping efficiency and compute utilization. It also tracks read and write requests
for IFMAP, Filter, and OFMAP operations.
"""

import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg
skew_factor = 3

class systolic_compute_os_sa:
    """
    Class that computes the output using Output Stationary dataflow.
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        # Params set by user
        self.config = cfg()

        self.ifmap_op_mat = np.zeros((1, 1))
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mat = np.zeros((1, 1))

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.arr_row = 0
        self.arr_col = 0

        self.row_fold = 1
        self.col_fold = 1

        # Generated matrices
        self.ifmap_op_mat_trans = np.zeros((1,1))
        self.ifmap_prefetch_matrix = np.zeros((1,1))
        self.filter_prefetch_matrix = np.zeros((1,1))

        self.ifmap_demand_matrix = np.zeros((1,1))
        self.ofmap_demand_matrix = np.zeros((1,1))
        self.filter_demand_matrix = np.zeros((1,1))

        # Generated metrics
        self.ifmap_reads = 0
        self.filter_reads = 0
        self.ofmap_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []
        self.total_cycles = None

        # Flags
        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

    #
    def set_params(self,
                   config_obj=cfg(),
                   ifmap_op_mat = np.zeros((1,1)),
                   ofmap_op_mat = np.zeros((1,1)),
                   filter_op_mat = np.zeros((1,1))
                ):
        """
        Method to set the output stationary run parameters for housekeeping.
        """

        self.config = config_obj
        self.arr_row, self.arr_col = self.config.get_array_dims()

        # mod global-fold-interleaving
        M, K = ifmap_op_mat.shape
        N = ofmap_op_mat.shape[1]
        
        F_M = math.ceil(M / self.arr_row)
        F_N = math.ceil(N / self.arr_col)
        
        target_M = F_M * self.arr_row
        target_N = F_N * self.arr_col
        
        # Padding
        if target_M > M:
            ifmap_op_mat = np.vstack((ifmap_op_mat, np.zeros((target_M - M, K))))
        if target_N > N:
            filter_op_mat = np.hstack((filter_op_mat, np.zeros((K, target_N - N))))
        
        new_ofmap = np.zeros((target_M, target_N))
        new_ofmap[:M, :N] = ofmap_op_mat
        ofmap_op_mat = new_ofmap

        # Prepare fold pairs
        fold_pairs = []
        for fc_idx in range(F_N):
            for fr_idx in range(F_M):
                fold_pairs.append((fr_idx, fc_idx))
        
        num_logical_folds = len(fold_pairs)
        num_physical_passes = math.ceil(num_logical_folds / skew_factor)
        
        while len(fold_pairs) < num_physical_passes * skew_factor:
            fold_pairs.append((-1, -1))
            
        # Construct unrolled and interleaved matrices
        self.ifmap_op_mat = np.full((num_physical_passes * self.arr_row, K * skew_factor), -1.0)
        self.filter_op_mat = np.full((K * skew_factor, num_physical_passes * self.arr_col), -1.0)
        self.ofmap_op_mat_full = np.full((num_physical_passes * self.arr_row * skew_factor, self.arr_col), -1.0)
        
        self.fold_pairs = fold_pairs # for compute util accuracy
        self.orig_K = K

        for p in range(num_physical_passes):
            for s in range(skew_factor):
                fr, fc = fold_pairs[p * skew_factor + s]
                if fr == -1: continue
                
                # IFMAP block
                i_start, i_end = fr * self.arr_row, (fr + 1) * self.arr_row
                self.ifmap_op_mat[p*self.arr_row : (p+1)*self.arr_row, s::skew_factor] = ifmap_op_mat[i_start:i_end, :]
                
                # Filter block
                f_start, f_end = fc * self.arr_col, (fc + 1) * self.arr_col
                self.filter_op_mat[s::skew_factor, p*self.arr_col : (p+1)*self.arr_col] = filter_op_mat[:, f_start:f_end]
                
                # OFMAP blocks (stored for sequential draining)
                o_block = ofmap_op_mat[i_start:i_end, f_start:f_end]
                for r in range(self.arr_row):
                    self.ofmap_op_mat_full[p * self.arr_row * skew_factor + r * skew_factor + s, :] = o_block[r, :]

        self.ofmap_op_mat = ofmap_op_mat[:self.ifmap_op_mat.shape[0], :self.arr_col]
        
        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row= self.filter_op_mat.shape[0]

        assert ifmap_col == filter_row, "Dimension mismatch between operands"
        self.ifmap_op_mat_trans = np.transpose(self.ifmap_op_mat)

        self.Sr = self.ifmap_op_mat.shape[0]
        self.Sc = self.filter_op_mat.shape[1]
        self.T = self.ifmap_op_mat.shape[1]

        self.row_fold = num_physical_passes
        self.col_fold = 1
        # mod end

        self.params_set_flag = True

    #
    def create_prefetch_matrices(self):
        """
        Method to create ifmap and filter prefetch matrices. These matrices are prefetched in the
        SRAM before running memory simulation.
        """
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_prefetch_mat()
        self.create_filter_prefetch_mat()

        self.prefetch_mat_ready_flag = True

    #
    def create_ifmap_prefetch_mat(self):
        """
        Method to create ifmap prefetch matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        for fr in range(self.row_fold):
            start_row_idx = fr * self.arr_row
            end_row_idx = min(start_row_idx + self.arr_row, self.Sr)

            delta = self.arr_row - (end_row_idx - start_row_idx)

            # The usage of row idx in cols is correct as this is the transposed matrix
            # Thus, Sr is along the cols and T is along the rows of this matrix
            # This is how the traces will be generated as well
            this_fold_prefetch = self.ifmap_op_mat_trans[:,start_row_idx: end_row_idx]

            #If there is under utilization, fill them with null requests
            if delta > 0:
                null_req_mat = np.ones((self.T, delta)) * -1
                this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

            if fr == 0:
                self.ifmap_prefetch_matrix = this_fold_prefetch
            else:
                self.ifmap_prefetch_matrix = \
                    np.concatenate((self.ifmap_prefetch_matrix, this_fold_prefetch), axis=0)

        M, N = self.ifmap_prefetch_matrix.shape
        num_elems = M * N
        num_diags = M + N
        prefetches = np.zeros((1,num_elems))
        idx = 0

        pbar = tqdm(total=M*N, disable=True)

        for diag_id in range(num_diags):
            max_row_id = min(diag_id, M - 1)
            min_row_id = max(0, diag_id - N + 1)
            valid_rows = max_row_id - min_row_id + 1

            for offset in range(valid_rows):
                row_id = max_row_id - offset
                col_id = diag_id - row_id

                elem = self.ifmap_prefetch_matrix[row_id][col_id]
                prefetches[0, idx] = elem
                idx += 1
                pbar.update(1)

        pbar.close()
        self.ifmap_prefetch_matrix = prefetches

    #
    def create_filter_prefetch_mat(self):
        """
        Method to create filter prefetch matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        for fc in range(self.row_fold):
            col_start_id = fc * self.arr_col
            col_end_id = col_start_id + self.arr_col

            delta = self.arr_col - (col_end_id - col_start_id)

            this_fold_prefetch = self.filter_op_mat[:,col_start_id:col_end_id]

            if delta > 0:
                null_req_mat = np.ones((self.T, delta)) * -1
                this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

            if fc == 0:
                self.filter_prefetch_matrix = this_fold_prefetch
            else:
                self.filter_prefetch_matrix = \
                    np.concatenate((self.filter_prefetch_matrix, this_fold_prefetch), axis=0)

        M, N = self.filter_prefetch_matrix.shape
        num_elems = M * N
        num_diags = M + N
        prefetches = np.zeros((1, num_elems))
        idx = 0

        pbar = tqdm(total=M * N, disable=True)

        for diag_id in range(num_diags):
            max_row_id = min(diag_id, M - 1)
            min_row_id = max(0, diag_id - N + 1)
            valid_rows = max_row_id - min_row_id + 1

            for offset in range(valid_rows):
                row_id = max_row_id - offset
                col_id = diag_id - row_id

                elem = self.filter_prefetch_matrix[row_id][col_id]
                prefetches[0, idx] = elem
                idx += 1
                pbar.update(1)

        pbar.close()
        self.filter_prefetch_matrix = prefetches

    #
    def create_demand_matrices(self):
        """
        Method to create ifmap, filter and ofmap demand matrices.
        """
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()

        maxT = max(self.ifmap_demand_matrix.shape[0],
                   self.filter_demand_matrix.shape[0],
                   self.ofmap_demand_matrix.shape[0])
        def pad_rows(mat, target):
            if mat.shape[0] == target:
                return mat
            pad = np.ones((target - mat.shape[0], mat.shape[1]),
                          dtype=mat.dtype) * -1
            return np.vstack([mat, pad])
        self.ifmap_demand_matrix = pad_rows(self.ifmap_demand_matrix, maxT)
        self.filter_demand_matrix = pad_rows(self.filter_demand_matrix, maxT)
        self.ofmap_demand_matrix  = pad_rows(self.ofmap_demand_matrix,  maxT)

        self.demand_mat_ready_flag = True

    #
    def create_ifmap_demand_mat(self):
        """
        Method to create ifmap demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (row_end_idx - row_start_id)

                this_fold_demand = self.ifmap_op_mat_trans[:,row_start_id: row_end_idx]
                self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                if delta > 0:
                    null_req_mat = np.ones((self.T, delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                this_fold_demand = skew_matrix(this_fold_demand)

                overlap = this_fold_demand.shape[1] - 1 
                if fr == 0 and fc == 0:
                    self.ifmap_demand_matrix = this_fold_demand
                else:
                    self.ifmap_demand_matrix = overlap_concat(
                        self.ifmap_demand_matrix,
                        this_fold_demand,
                        overlap
                    )

                pbar.update(1)

        pbar.close()

    #
    def create_filter_demand_mat(self):
        """
        Method to create filter demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fr * self.arr_col
                col_end_idx = col_start_id + self.arr_col
                delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.filter_op_mat[:, col_start_id: col_end_idx]
                self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                if delta > 0:
                    null_req_mat = np.ones((self.T, delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                this_fold_demand = skew_matrix(this_fold_demand)
                overlap = this_fold_demand.shape[1] - 1 

                if fr == 0 and fc == 0:
                    self.filter_demand_matrix = this_fold_demand
                else:
                    self.filter_demand_matrix = overlap_concat(
                        self.filter_demand_matrix,
                        this_fold_demand,
                        overlap
                    )

                pbar.update(1)

        pbar.close()

    #
    def create_ofmap_demand_mat(self):
        """
        Method to create ofmap demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = self.T - 1
        inter_fold_gap_prefix_mat = np.ones((max(inter_fold_gap_prefix, 0), self.arr_col)) * -1
        inter_fold_gap_between = self.T - (skew_factor * self.arr_row) - (self.arr_col - 1)
        inter_fold_gap_between_mat = np.ones((max(inter_fold_gap_between, 0), self.arr_col)) * -1

        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                col_start_id = 0
                col_end_idx = self.arr_col

                logical_start = row_start_id * skew_factor
                logical_end = row_end_idx * skew_factor
                
                block = self.ofmap_op_mat_full[logical_start:logical_end, col_start_id:col_end_idx]
                num_phy_rows = row_end_idx - row_start_id
                block_reshaped = block.reshape(num_phy_rows, skew_factor, -1)
                block_reversed = block_reshaped[:, ::-1, :]
                this_fold_demand = block_reversed.reshape(-1, block_reversed.shape[-1])

                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                if row_delta > 0:
                    null_req_mat = np.ones((row_delta * skew_factor, self.arr_col)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                this_fold_demand = np.flip(this_fold_demand, 0)

                if fr == 0 and fc == 0:
                    this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand),
                                                      axis=0)
                else:
                    this_fold_demand = np.concatenate((inter_fold_gap_between_mat, this_fold_demand),
                                                      axis=0)

                row_used = min(self.arr_row, row_end_idx - row_start_id)
                col_used = self.arr_col
                mac_used = row_used * col_used
                mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)

                cycles_this_fold = this_fold_demand.shape[0] + this_fold_demand.shape[1] - 1
                
                valid_folds = 0
                for s_idx in range(skew_factor):
                    if self.fold_pairs[fr * skew_factor + s_idx][0] != -1:
                        valid_folds += 1
                compute_cycles_this_fold = mac_used * self.orig_K * valid_folds
                
                compute_util_this_fold = \
                    compute_cycles_this_fold / (self.arr_row * self.arr_col * cycles_this_fold)

                self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
                self.compute_utility_per_fold.append(compute_util_this_fold)

                this_fold_demand = skew_matrix(this_fold_demand)

                if fr == 0 and fc == 0:
                    self.ofmap_demand_matrix = this_fold_demand
                else:
                    self.ofmap_demand_matrix = np.concatenate(
                        (self.ofmap_demand_matrix, this_fold_demand), axis=0
                    )

                pbar.update(1)

        pbar.close()

    #
    def get_ifmap_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()
        return self.ifmap_prefetch_matrix

    def get_filter_prefetch_mat(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()
        return self.filter_prefetch_matrix

    def get_prefetch_matrices(self):
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()
        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    def get_ifmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()
        return self.ifmap_demand_matrix

    def get_filter_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()
        return self.filter_demand_matrix

    def get_ofmap_demand_mat(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()
        return self.ofmap_demand_matrix

    def get_demand_matrices(self):
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()
        return self.ifmap_demand_matrix, self.filter_demand_matrix, self.ofmap_demand_matrix

    def get_avg_mapping_efficiency(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)
        return agg / num

    def get_avg_compute_utilization(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        if self.total_cycles is not None:
            total_compute_cycles = self.Sr * self.Sc * self.T # Simplified for unrolled
            return total_compute_cycles / (self.arr_row * self.arr_col * self.total_cycles)
        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)
        return agg / num

    def set_total_cycles(self, total_cycles):
        self.total_cycles = total_cycles

    def get_ifmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ifmap_reads

    def get_filter_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.filter_reads

    def get_ofmap_requests(self):
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ofmap_writes

#
def skew_matrix(input_matrix_np):
    rows, cols = input_matrix_np.shape
    out_matrix_np = np.full((rows + cols - 1, cols), -1, dtype=input_matrix_np.dtype)
    for c in range(cols):
        out_matrix_np[c:c + rows, c] = input_matrix_np[:, c]
    return out_matrix_np

def overlap_concat(prev, nxt, overlap):
    if prev is None:
        return nxt
    assert prev.shape[1] == nxt.shape[1]
    tail = prev[-overlap:, :]
    head = nxt[:overlap, :]
    merged = np.where(head != -1, head, tail)
    return np.vstack([prev[:-overlap, :], merged, nxt[overlap:, :]])
