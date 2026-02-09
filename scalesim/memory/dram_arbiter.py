"""
Global DRAM arbiter that enforces a total bandwidth cap across multiple ports.
"""

import math


class DramArbiter:
    """
    Simple FCFS arbiter that serializes requests to enforce total bandwidth.
    """

    def __init__(self, total_bw_words_per_cycle=0, priority_gap_cycles=1):
        """
        Initialize the arbiter with a total bandwidth cap in words/cycle.
        """
        self.total_bw = int(total_bw_words_per_cycle)
        self.priority_gap = int(priority_gap_cycles)  # NOTE: Extra cycles to deprioritize low-priority traffic
        self.current_cycle = 0  # NOTE: Global arbitration timeline (cycle index)
        self.remaining_bw = int(total_bw_words_per_cycle)  # NOTE: Remaining words for current cycle
        self.trace_records = []  # NOTE: Per-request scheduling trace

    def reset(self):
        """
        Reset internal scheduling state.
        """
        self.current_cycle = 0  # NOTE: Reset arbitration timeline
        self.remaining_bw = int(self.total_bw)  # NOTE: Reset remaining bandwidth
        self.trace_records = []  # NOTE: Clear trace on reset

    def service_requests(self, arrival_cycles, request_sizes, extra_latency=0, priorities=None, sources=None):
        """
        Service requests in FCFS order using a single shared bandwidth budget.
        Returns completion cycles for each request.
        """
        if priorities is None:
            priorities = [0 for _ in arrival_cycles]
        if sources is None:
            sources = ["unknown" for _ in arrival_cycles]
        out_cycles = []

        for arrival, size, prio, src in zip(arrival_cycles, request_sizes, priorities, sources):
            if size <= 0 or self.total_bw <= 0:
                out_cycles.append(int(arrival) + int(extra_latency))
                continue

            effective_arrival = int(arrival) + int(prio) * self.priority_gap  # NOTE: Lower priority arrives later
            if self.current_cycle < int(effective_arrival):
                self.current_cycle = int(effective_arrival)
                self.remaining_bw = int(self.total_bw)

            start_cycle = int(self.current_cycle)
            remaining_size = int(size)

            # NOTE: Allocate bandwidth across cycles to allow packing multiple requests per cycle
            while remaining_size > self.remaining_bw:
                remaining_size -= self.remaining_bw
                self.current_cycle += 1
                self.remaining_bw = int(self.total_bw)

            self.remaining_bw -= remaining_size
            end_cycle = int(self.current_cycle)
            completion = end_cycle + int(extra_latency)
            out_cycles.append(completion)
            wait_cycles = max(start_cycle - int(arrival), 0)
            self.trace_records.append(  # NOTE: Store per-request arbitration outcome
                (str(src), int(arrival), int(start_cycle), int(completion), int(size), int(wait_cycles), int(prio))
            )

        return out_cycles

    def get_trace_records(self):
        """
        Method to get the per-request arbitration trace.
        """
        return list(self.trace_records)  # NOTE: Return a copy of the trace

    def get_summary(self):
        """
        Method to get summary statistics grouped by source.
        """
        summary = {}
        for src, arrival, start, end, size, wait, prio in self.trace_records:
            if src not in summary:
                summary[src] = {"requests": 0, "total_words": 0, "total_wait": 0, "max_wait": 0}
            summary[src]["requests"] += 1
            summary[src]["total_words"] += int(size)
            summary[src]["total_wait"] += int(wait)
            summary[src]["max_wait"] = max(summary[src]["max_wait"], int(wait))
        return summary
