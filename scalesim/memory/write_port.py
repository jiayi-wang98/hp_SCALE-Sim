"""
External DRAM write requests serviced by Ramulator
"""
import numpy as np
from scalesim.scale_config import scale_config as config
from bisect import bisect_left
from scalesim.memory.dram_arbiter import DramArbiter

# This is shell module to ensure continuity

class write_port:
    """
    Class to define external DRAM memory requests with Ramulator
    """
    #
    def __init__(self):
        """
        __init__ module.
        """
        self.latency = 0
        self.ramulator_trace = False
        self.latency_matrix = []
        self.bw = 10
        self.request_queue_size = 100
        self.request_queue_status = 0
        self.stall_cycles = 0
        self.request_array = []
        self.count = 0
        self.config = config()
        self.arbiter = None  # NOTE: Optional global DRAM arbiter
        self.port_name = "unknown"  # NOTE: Port identifier for arbitration logs
    
    def def_params( self,
                    config = config(),
                    latency_file =''
                ):
        """
        Method to define the paths of ramulator trace numpy files 
        and write request queue sizes.
        """
        self.config = config
        self.ramulator_trace = self.config.get_ramulator_trace()
        self.request_queue_size = self.config.get_req_buf_sz_wr()
        self.bw = self.config.get_bandwidths_as_list()[0]
        if self.ramulator_trace == True:
            self.latency_matrix = np.load(latency_file)
        self.latency=0
    #

    def set_arbiter(self, arbiter, port_name="unknown"):
        """
        Method to attach a global DRAM arbiter.
        """
        self.arbiter = arbiter  # NOTE: Shared arbiter enforces total DRAM bandwidth
        self.port_name = str(port_name)  # NOTE: Tag requests with a human-readable source name

    def find_latency(self):
        """
        Method to map DRAM return path latency for each transactions.
        """
        if(self.count < len(self.latency_matrix)):
            latency_out = self.latency_matrix[self.count]
            #print(str(self.count)+ ' ' + str(latency_out))
            self.count+=1
        else:
            latency_out = self.latency
        if(latency_out > 10000):
            latency_out = 0

        return latency_out

    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        """
        Method to service read request by the read buffer.
        Check for hit in the request queue or add the DRAM
        roundtrip latency for each transaction reported by 
        Ramulator.
        """
        if self.ramulator_trace == False:
            if self.arbiter is not None and isinstance(self.arbiter, DramArbiter):
                arrival_cycles = [int(x[0]) for x in incoming_cycles_arr_np]
                request_sizes = [int(np.sum(row != -1)) for row in incoming_requests_arr_np]
                priorities = [1 for _ in arrival_cycles]  # NOTE: Writes are lower priority
                sources = [self.port_name for _ in arrival_cycles]  # NOTE: Tag source for trace logging
                out_cycles = self.arbiter.service_requests(arrival_cycles, request_sizes, extra_latency=self.latency, priorities=priorities, sources=sources)
                return np.asarray(out_cycles).reshape((len(out_cycles), 1))  # NOTE: FCFS DRAM arbitration
            out_cycles_arr_np = incoming_cycles_arr_np + self.latency
            out_cycles_arr_np = out_cycles_arr_np.reshape((out_cycles_arr_np.shape[0], 1))
            return out_cycles_arr_np

        updated_req_timestamp = incoming_cycles_arr_np[0][0]
        print(updated_req_timestamp)
        out_cycles_arr = np.zeros(incoming_cycles_arr_np.shape[0])
        for i in range(len(incoming_cycles_arr_np)):
            out_cycles_arr[i] = incoming_cycles_arr_np[i][0] + self.stall_cycles + self.find_latency()
            self.request_array.append(out_cycles_arr[i])
            if len(self.request_array) == self.request_queue_size:
                updated_req_timestamp = incoming_cycles_arr_np[i][0] + self.stall_cycles
                self.request_array.sort()
                if self.request_array[0] >= updated_req_timestamp:
                    self.stall_cycles += self.request_array[0] - updated_req_timestamp
                    updated_req_timestamp = self.request_array[0]
                    self.request_array.pop(0)
                else:
                    index = bisect_left(self.request_array,updated_req_timestamp)
                    if index == len(self.request_array):
                        self.request_array = []
                        print("Empty array")
                    else:
                        self.request_array = self.request_array[index:]
            elif len(self.request_array) > self.request_queue_size:
                self.request_array = self.request_array[-self.request_queue_size:]
        # --- Placeholder for zeroization logic
        #if flush == 0:
        #    self.stall_cycles = 0

        self.stall_cycles =0
        return out_cycles_arr
