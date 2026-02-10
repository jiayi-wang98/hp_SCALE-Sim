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
skew_factor = 2

class systolic_compute_os_sa_f2:
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

        # mod for unrolled prefetch
        for fc in range(self.row_fold):
            col_start_id = fc * self.arr_col
            col_end_id = col_start_id + self.arr_col
            # mod end

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
        print("[DBG os sa f2]")
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

        # mod for pre and suf fix
        # inter_fold_gap_suffix = self.arr_row - 1
        # inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

        # Debug messages
        #print('DEBUG: create_filter_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                # mod for unrolled filter
                col_start_id = fr * self.arr_col
                col_end_idx = col_start_id + self.arr_col
                # mod end
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

        # mod for of map counting
        inter_fold_gap_prefix = self.T - 1
        inter_fold_gap_prefix_mat = np.ones((max(inter_fold_gap_prefix, 0), self.arr_col)) * -1
        inter_fold_gap_between = self.T - (skew_factor * self.arr_row) - (self.arr_col - 1)
        inter_fold_gap_between_mat = np.ones((max(inter_fold_gap_between, 0), self.arr_col)) * -1

        # Debug messages
        #print('DEBUG: create_ifmap_demand_mat()')
        pbar = tqdm(total=self.col_fold * self.row_fold, disable=True)

        for fc in range(self.col_fold):
            for fr in range(self.row_fold):
                row_start_id = fr * self.arr_row
                row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                # mod for unrolled ofmap
                col_start_id = 0
                col_end_idx = self.arr_col
                col_delta = 0 
                # mod end

                # --- START: Modified for 3-Cycle Output ---
                # We need to extract the full 3x rows corresponding to these physical rows
                logical_start = row_start_id * skew_factor
                logical_end = row_end_idx * skew_factor
                
                # Extract block: (NumPhy * 3, Cols)
                block = self.ofmap_op_mat_full[logical_start:logical_end, col_start_id:col_end_idx]
                
                # Reshape to (NumPhy, 3, Cols) to manipulate the temporal batch
                num_phy_rows = row_end_idx - row_start_id
                # Note: block.shape[0] should be num_phy_rows * 3
                
                # We want to stack them such that after 'flip' (reflection along rows), they come out in correct order.
                # Standard 'flip' reverses the 0-th dimension.
                # If we want A, B, C to come out in order 0, 1, 2...
                # And 'skew_matrix' drains Row 0 first?
                # Usually systolic: Bottom physical row drains first.
                # Within a physical row, we have temporal sequence A->B->C.
                # So we want sequence A, B, C.
                # If we produce a stack [A, B, C], and 'flip' reverses it to [C, B, A].
                # Then 'skew_matrix' (if it drains top-down) would drain C, then B, then A.
                # To get A, B, C, we need to present [A, B, C] to skew_matrix.
                # So we need [C, B, A] BEFORE flip.
                
                block_reshaped = block.reshape(num_phy_rows, skew_factor, -1)
                
                # Reverse the inner batch dimension: [A, B, C] -> [C, B, A]
                block_reversed = block_reshaped[:, ::-1, :]
                
                # Flatten back to (NumPhy * 3, Cols)
                this_fold_demand = block_reversed.reshape(-1, block_reversed.shape[-1])
                # --- END: Modified ---

                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few
                # rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                if row_delta > 0:
                    # Note: We are padding PHYSICAL rows.
                    # So we need to add 'row_delta * 3' null rows?
                    # Yes, to maintain the time structure.
                    null_req_mat = np.ones((row_delta * skew_factor, self.arr_col)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                # Reflect along the rows
                # This is a characteristic of the fact that the outputs are streamed out from the
                # bottom edge.
                # If the outputs are streamed out from the top edge instead, then this step is not
                # needed.
                this_fold_demand = np.flip(this_fold_demand, 0)
                self.ofmap_writes += this_fold_demand.shape[0] + this_fold_demand.shape[1]

                # Now add the prefix/gap matrix (before skew)
                # Prefix accounts for the initial T-1 latency. Gaps align subsequent folds.
                if fr == 0 and fc == 0:
                    this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand),
                                                      axis=0)
                else:
                    this_fold_demand = np.concatenate((inter_fold_gap_between_mat, this_fold_demand),
                                                      axis=0)

                # Calculate the mapping efficiency
                row_used = min(self.arr_row, row_end_idx - row_start_id)
                col_used = self.arr_col 
                mac_used = row_used * col_used
                mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)

                cycles_this_fold = this_fold_demand.shape[0] + this_fold_demand.shape[1] - 1
                
                # mod accuracy
                valid_folds = 0
                for s_idx in range(skew_factor):
                    if self.fold_pairs[fr * skew_factor + s_idx][0] != -1:
                        valid_folds += 1
                compute_cycles_this_fold = mac_used * self.orig_K * valid_folds
                # mod end
                
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


                # Keep explicit gaps between folds; no overlap merge.
                if fr == 0 and fc == 0:
                    self.ofmap_demand_matrix = this_fold_demand
                else:
                    self.ofmap_demand_matrix = np.concatenate(
                        (self.ofmap_demand_matrix, this_fold_demand), axis=0
                    )


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

        # mod start compute util accuracy
        if self.total_cycles is not None:
            valid_fold_count = sum(1 for pair in self.fold_pairs if pair[0] != -1)
            total_compute_cycles = valid_fold_count * self.arr_row * self.arr_col * self.orig_K
            avg_compute_util = \
                total_compute_cycles / (self.arr_row * self.arr_col * self.total_cycles)
            return avg_compute_util
        # mod end compute util accuracy

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        avg_compute_util = agg / num

        return avg_compute_util

    # mod start compute util
    def set_total_cycles(self, total_cycles):
        """
        Method to set total cycles for compute utilization.
        """
        self.total_cycles = total_cycles
    # mod end compute util

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