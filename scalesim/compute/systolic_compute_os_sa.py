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


        # --- START: Modified for Pipelined 3-Cycle MAC Interleaving ---
        # The goal is to interleave 3 rows of IFMAP into the time dimension (columns)
        # to hide the 3-cycle MAC latency.
        
        # 1. Handle IFMAP: Interleave 3 sub-matrices
        M, K = ifmap_op_mat.shape
        
        # Pad M (rows) if not divisible by 3 to ensure we can split evenly
        remainder = M % skew_factor
        if remainder != 0:
            pad_rows = skew_factor - remainder
            padding = np.zeros((pad_rows, K))
            ifmap_op_mat = np.vstack((ifmap_op_mat, padding))
            # If we pad IFMAP, we must logically consider OFMAP padded too for dimension consistency
            # though we will resize OFMAP below anyway.
            M += pad_rows

        # Reshape to separate the groups: (M/3, 3, K)
        # Dimension 0: The new reduced spatial rows
        # Dimension 1: The 3 interleaved batches
        # Dimension 2: The original time/reduction dimension
        ifmap_reshaped = ifmap_op_mat.reshape(M // skew_factor, skew_factor, K)
        
        # Permute to (M/3, K, 3) to put the interleaved batches adjacent to each other
        ifmap_permuted = ifmap_reshaped.transpose(0, 2, 1)
        
        # Flatten the last two dimensions to create the new interleaved time dimension
        # New shape: (M/3, 3*K)
        # The sequence in columns will be: [K0_Mat0, K0_Mat1, K0_Mat2, K1_Mat0, K1_Mat1, K1_Mat2...]
        self.ifmap_op_mat = ifmap_permuted.reshape(M // skew_factor, -1)

        # 2. Handle Filter: Repeat weights to match the interleaved input
        # Original: (K, N). Target: (3*K, N)
        # We repeat each row 3 times: [W0, W0, W0, W1, W1, W1...]
        self.filter_op_mat = np.repeat(filter_op_mat, skew_factor, axis=0)

        # 3. Handle OFMAP: Adjust spatial dimension
        # Original: (M, N). Target: (M/3, N)
        # Since we compressed 3 spatial rows into 1 pipeline stream, the output matrix 
        # seen by the simulator (for demand generation) assumes fewer rows.
        # Note: In a real value simulation, this would require unpacking, but for 
        # cycle/trace generation, resizing is sufficient.
        self.ofmap_op_mat = ofmap_op_mat[:(M // skew_factor), :]
        
        # self.ifmap_op_mat = ifmap_op_mat
        # self.filter_op_mat = filter_op_mat
        # self.ofmap_op_mat = ofmap_op_mat

        # --- END: Modified for Pipelined 3-Cycle MAC Interleaving ---

        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row= self.filter_op_mat.shape[0]

        assert ifmap_col == filter_row, "Dimension mismatch between operands"
        self.ifmap_op_mat_trans = np.transpose(self.ifmap_op_mat)

        self.Sr = self.ifmap_op_mat.shape[0]
        self.Sc = self.filter_op_mat.shape[1]
        self.T = self.ifmap_op_mat.shape[1]

        self.arr_row, self.arr_col = self.config.get_array_dims()

        self.row_fold = math.ceil(self.Sr / self.arr_row)
        self.col_fold = math.ceil(self.Sc / self.arr_col)

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

        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a
        # skew in demand
        #print('DEBUG: create_ifmap_prefetch_mat()')
        #start_time = time.time()

        M, N = self.ifmap_prefetch_matrix.shape
        num_elems = M * N
        num_diags = M + N
        prefetches = np.zeros((1,num_elems))
        idx = 0

        pbar = tqdm(total=M*N, disable=True)
        #print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

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

        #t = time.time() - start_time
        #print('DEBUG: create_ifmap_prefetch_mat =' + str(t))

    #
    def create_filter_prefetch_mat(self):
        """
        Method to create filter prefetch matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        for fc in range(self.col_fold):
            col_start_id = fc * self.arr_col
            col_end_id = min(col_start_id + self.arr_col, self.Sc)

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

        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a
        # skew in demand
        #print('DEBUG: create_filter_prefetch_mat()')
        #start_time = time.time()

        M, N = self.filter_prefetch_matrix.shape
        num_elems = M * N
        num_diags = M + N
        prefetches = np.zeros((1, num_elems))
        idx = 0

        pbar = tqdm(total=M * N, disable=True)
        # print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

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

        #t = time.time() - start_time
        #print('DEBUG: create_filter_prefetch_mat =' + str(t))

    #
    def create_demand_matrices(self):
        # debug
        print("[DBG os sa]")
        """
        Method to create ifmap, filter and ofmap demand matrices from the operand matrices. They
        contain several folds of ifmap, filter and ofmap demands. The folding happens because
        operand matrices are generally larger than systolic array dimensions.
        """
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()

        # mod for overlap, test
        print("[DBG pre-pad shapes]",
            self.ifmap_demand_matrix.shape,
            self.filter_demand_matrix.shape,
            self.ofmap_demand_matrix.shape)


        #mod for overlap
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

        # mod for overlap, test
        print("[DBG post-pad shapes]",
            self.ifmap_demand_matrix.shape,
            self.filter_demand_matrix.shape,
            self.ofmap_demand_matrix.shape,
            "maxT=", maxT)



        assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], \
               'IFMAP and Filter demands out of sync'
        assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], \
               'OFMAP and Filter demands out of sync'
        assert self.ifmap_demand_matrix.shape[1] == self.arr_row, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.arr_col,'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'
        
        # mod for overlap, test
        print(f"[DBG] Sr={self.Sr} Sc={self.Sc} T={self.T} "
            f"arr={self.arr_row}x{self.arr_col} "
            f"row_fold={self.row_fold} col_fold={self.col_fold}")


        self.demand_mat_ready_flag = True

    #
    def create_ifmap_demand_mat(self):
        """
        Method to create ifmap demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        # Anand: Concatenation issue fix
        inter_fold_gap_suffix = self.arr_col - 1
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        # DEBUG section
        #print('DEBUG: create_ifmap_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                delta = self.arr_row - (row_end_idx - row_start_id)

                # Indexing the cols with row start and row end idx are correct
                # See the comment on ifmap_prefetch generation
                this_fold_demand = self.ifmap_op_mat_trans[:,row_start_id: row_end_idx]
                self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((self.T, delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # In this computation scheme we are allowing the generated outputs to drain out
                # before starting the next fold
                # This portion accounts for that extra time by adding null requests
                # mod for overlap
                # this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat),
                #                                  axis=0)

                # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                # if fr == 0 and fc == 0:
                #     self.ifmap_demand_matrix = this_fold_demand
                # else:
                #     self.ifmap_demand_matrix = \
                #         np.concatenate((self.ifmap_demand_matrix, this_fold_demand), axis=0)

                # mod for overlap
                overlap = this_fold_demand.shape[1] - 1   # cols - 1 = arr_row - 1 （因为ifmap_demand列数是arr_row）
                if fr == 0 and fc == 0:
                    self.ifmap_demand_matrix = this_fold_demand
                else:
                    self.ifmap_demand_matrix = overlap_concat(self.ifmap_demand_matrix,
                                                            this_fold_demand,
                                                            overlap)

                pbar.update(1)

        pbar.close()
        # TODO: cleanup
        # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
        #self.ifmap_demand_matrix = skew_matrix(self.ifmap_demand_matrix)

    #
    def create_filter_demand_mat(self):
        """
        Method to create filter demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_suffix = self.arr_row - 1
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

        # Debug messages
        #print('DEBUG: create_filter_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.filter_op_mat[:, col_start_id: col_end_idx]
                self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account under utilization
                if delta > 0:
                    null_req_mat = np.ones((self.T, delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # In this computation scheme we are allowing the generated outputs to drain out
                # before starting the next fold
                # This portion accounts for that extra time by adding null requests
                # mod for overlap
                # this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat),
                #                                  axis=0)

                # Add skew to the Filter demand matrix to reflect systolic pipeline fill
                # mod for overlap
                # this_fold_demand = skew_matrix(this_fold_demand)

                # mod for overlap
                this_fold_demand = skew_matrix(this_fold_demand)
                overlap = this_fold_demand.shape[1] - 1  # cols - 1

                # mod for overlap
                # if fr == 0 and fc == 0:
                #     self.filter_demand_matrix = this_fold_demand
                # else:
                #     self.filter_demand_matrix = \
                #         np.concatenate((self.filter_demand_matrix, this_fold_demand), axis=0)

                overlap = this_fold_demand.shape[1] - 1  # cols - 1 = arr_col - 1 （因为filter_demand列数是arr_col）
                if fr == 0 and fc == 0:
                    self.filter_demand_matrix = this_fold_demand
                else:
                    self.filter_demand_matrix = overlap_concat(self.filter_demand_matrix,
                                                            this_fold_demand,
                                                            overlap)


                pbar.update(1)

        pbar.close()
        # TODO: Cleanup
        # Add skew to the Filter demand matrix to reflect systolic pipeline fill
        #self.filter_demand_matrix = skew_matrix(self.filter_demand_matrix)

    #
    def create_ofmap_demand_mat(self):
        """
        Method to create ofmap demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = self.T  - 1
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_col)) * -1

        # Debug messages
        #print('DEBUG: create_ifmap_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = \
                    self.ofmap_op_mat[row_start_id: row_end_idx, col_start_id: col_end_idx]
                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few
                # rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                if row_delta > 0:
                    null_req_mat = np.ones((row_delta, self.arr_col)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                # Reflect along the rows
                # This is a characteristic of the fact that the outputs are streamed out from the
                # bottom edge.
                # If the outputs are streamed out from the top edge instead, then this step is not
                # needed.
                this_fold_demand = np.flip(this_fold_demand, 0)
                self.ofmap_writes += this_fold_demand.shape[0] + this_fold_demand.shape[1]

                # Now add the prefix matrix
                # These are the null demands to account for when the operands are streamed in
                # and the OFMAPS are not ready
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand),
                                                  axis=0)

                # Calculate the mapping efficiency
                row_used = min(self.arr_row, row_end_idx - row_start_id)
                col_used = min(self.arr_col, col_end_idx - col_start_id)
                mac_used = row_used * col_used
                mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)

                cycles_this_fold = this_fold_demand.shape[0] + this_fold_demand.shape[1] - 1
                compute_cycles_this_fold = mac_used * self.T
                compute_util_this_fold = \
                    compute_cycles_this_fold / (self.arr_row * self.arr_col * cycles_this_fold)

                self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
                self.compute_utility_per_fold.append(compute_util_this_fold)

                # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                # mod for overlap
                # if fr == 0 and fc == 0:
                #     self.ofmap_demand_matrix = this_fold_demand
                # else:
                #     self.ofmap_demand_matrix = \
                #         np.concatenate((self.ofmap_demand_matrix, this_fold_demand), axis=0)


                # mod for overlap
                overlap = this_fold_demand.shape[1] - 1   # = arr_col - 1，因为OFMAP demand的列数是arr_col
                if fr == 0 and fc == 0:
                    self.ofmap_demand_matrix = this_fold_demand
                else:
                    self.ofmap_demand_matrix = overlap_concat(self.ofmap_demand_matrix,
                                                            this_fold_demand,
                                                            overlap)


                pbar.update(1)

        pbar.close()
        # TODO: cleanup
        # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
        #self.ofmap_demand_matrix = skew_matrix(self.ofmap_demand_matrix)

    #
    def get_ifmap_prefetch_mat(self):
        """
        Method to get ifmap prefetch matrix.
        """
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix

    #
    def get_filter_prefetch_mat(self):
        """
        Method to get filter prefetch matrix.
        """
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.filter_prefetch_matrix

    #
    def get_prefetch_matrices(self):
        """
        Method to get ifmap and filter prefetch matrices.
        """
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    #
    def get_ifmap_demand_mat(self):
        """
        Method to get ifmap demand matrix.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix

    #
    def get_filter_demand_mat(self):
        """
        Method to get filter demand matrix.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.filter_demand_matrix

    #
    def get_ofmap_demand_mat(self):
        """
        Method to get ofmap demand matrix.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ofmap_demand_matrix

    #
    def get_demand_matrices(self):
        """
        Method to get ifmap, filter and ofmap demand matrices.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix, self.filter_demand_matrix, self.ofmap_demand_matrix

    #
    def get_avg_mapping_efficiency(self):
        """
        Method to get average mapping efficincy on the systolic array.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)

        avg_mapping_eff = agg / num

        return avg_mapping_eff

    #
    def get_avg_compute_utilization(self):
        """
        Method to get average compute utilization on the systolic array.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        avg_compute_util = agg / num

        return avg_compute_util

    #
    def get_ifmap_requests(self):
        """
        Method to get ifmap read requests.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ifmap_reads

    #
    def get_filter_requests(self):
        """
        Method to get filter read requests.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.filter_reads

    #
    def get_ofmap_requests(self):
        """
        Method to get ofmap write requests.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ofmap_writes

#
def skew_matrix(input_matrix_np):
    """
    Method to add skew to the input matix to maintain systolic array flow.
    Example:
        Input matrix:
        1 1 1 1 1 1 1 1 1

        Output matrix:
            1 1 1
          1 1 1
        1 1 1
    """
    rows, cols = input_matrix_np.shape

    out_matrix_np = np.full((rows + cols - 1, cols), -1, dtype=input_matrix_np.dtype)

    for c in range(cols):
        out_matrix_np[c:c + rows, c] = input_matrix_np[:, c]

    return out_matrix_np

def overlap_concat(prev, nxt, overlap):
    """
    prev, nxt: 2D numpy arrays with -1 as null.
    overlap: number of rows to overlap (typically cols - 1)
    """
    if prev is None:
        return nxt

    assert prev.shape[1] == nxt.shape[1]

    # split
    tail = prev[-overlap:, :]
    head = nxt[:overlap, :]

 
    merged = np.where(head != -1, head, tail)

    return np.vstack([prev[:-overlap, :], merged, nxt[overlap:, :]])

