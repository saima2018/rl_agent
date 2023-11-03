import math
import collections
import random
import numpy as np
import ipdb
from utils_project import schedule
#from dstructs import segtree
#from dstructs import segment_tree


class Transition(collections.namedtuple(
    "Transition",
    ('s_t', 'a_t', 'r_t', 's_t_1', 'buffer_idx')
    )):
    pass


class ReplayBuffer(object):
    def __init__(self, capacity, trans_mode="Transition"):
        print("Buffer Capacity=%d" % capacity)
        self._capacity = capacity
        self._memory = []
        self._stats = []
        self._next_idx = 0
        self._num_pos_trans = 0
        self._num_sample_called = 0
        # buffer stats 
        self._num_pos_trans_per_batch = 0
        self._num_neg_trans_per_batch = 0
        self._num_unique_pos_trans_per_batch = 0
        self._num_unique_neg_trans_per_batch = 0
        self._num_sample_since_last_get_stats = 0

    def push(self, *args):
        trans_args = args[:4]
        raw_r_t = trans_args[2].item()
        if len(self._memory) < self._capacity:
            self._memory.append(None)
            self._stats.append(None)
            if len(self._memory) == self._capacity:
                print("Replay Buffer memory MAX %d" % self._capacity)
        saved_idx = self._next_idx
        self._stats_update_push(raw_r_t, saved_idx)
        self._memory[saved_idx] = Transition(*trans_args[:4], saved_idx)
        # Move circular cursor
        self._next_idx = (self._next_idx + 1) % self._capacity
        return saved_idx

    def _stats_update_push(self, raw_r_t, saved_idx):
        was_pos_trans = ((self._stats[saved_idx] is not None) and self._stats[saved_idx][0])
        is_pos_trans = (raw_r_t > 0)
        if was_pos_trans:
            self._num_pos_trans -= 1
        if is_pos_trans:
            self._num_pos_trans += 1
        # [pos?, num_times_sampled, stored_at, last_err]
        self._stats[saved_idx] = [is_pos_trans, 0, self._num_sample_called]

    def sample(self, m):
        sampled_transitions = random.sample(self._memory, m)
        self._stats_update_sample(sampled_transitions)
        batch = Transition(*zip(*sampled_transitions))
        return batch

    def _stats_update_sample(self, sampled_transitions):
        self._num_sample_called += 1
        pos_buffer_idxs = []
        neg_buffer_idxs = []
        for trans in sampled_transitions:
            buffer_idx = trans.buffer_idx
            self._stats[buffer_idx][1] += 1
            if self._stats[buffer_idx][0]:
                self._num_pos_trans_per_batch += 1
                pos_buffer_idxs.append(buffer_idx)
            else:
                self._num_neg_trans_per_batch += 1
                neg_buffer_idxs.append(buffer_idx)
        self._num_unique_pos_trans_per_batch += len(set(pos_buffer_idxs))
        self._num_unique_neg_trans_per_batch += len(set(neg_buffer_idxs))
        self._num_sample_since_last_get_stats += 1

    def get_stats(self):
        """
        Get stats info for the transitions in current buffer
        (old buffers kept stats but were thrown away)
        """
        avg_times_each_trans_in_curr_buffer_sampled = 0
        avg_times_each_pos_trans_in_curr_buffer_sampled = 0
        avg_times_each_neg_trans_in_curr_buffer_sampled = 0
        avg_times_in_buffer = 0
        for stat in self._stats:
            avg_times_each_trans_in_curr_buffer_sampled += stat[1]
            if stat[0]:
                avg_times_each_pos_trans_in_curr_buffer_sampled += stat[1]
            else:
                avg_times_each_neg_trans_in_curr_buffer_sampled += stat[1]
            avg_times_in_buffer += (self._num_sample_called - stat[2])
        # ipdb.set_trace()
        buffer_size = float(len(self._stats))
        num_pos_trans = float(self._num_pos_trans)
        num_neg_trans = (buffer_size - num_pos_trans)
        avg_times_each_trans_in_curr_buffer_sampled /= buffer_size
        if num_pos_trans > 0:
            avg_times_each_pos_trans_in_curr_buffer_sampled /= num_pos_trans
        if num_neg_trans > 0:
            avg_times_each_neg_trans_in_curr_buffer_sampled /= num_neg_trans
        avg_times_in_buffer /= buffer_size

        num_sample_since_last = float(self._num_sample_since_last_get_stats)
        avg_num_pos_trans_per_batch = self._num_pos_trans_per_batch / num_sample_since_last
        avg_num_neg_trans_per_batch = self._num_neg_trans_per_batch / num_sample_since_last
        avg_num_unique_pos_trans_per_batch = self._num_unique_pos_trans_per_batch /num_sample_since_last
        avg_num_unique_neg_trans_per_batch = self._num_unique_neg_trans_per_batch /num_sample_since_last
        self._num_sample_since_last_get_stats = 0
        self._num_pos_trans_per_batch = 0
        self._num_neg_trans_per_batch = 0
        self._num_unique_pos_trans_per_batch = 0
        self._num_unique_neg_trans_per_batch = 0
        return  avg_times_in_buffer, avg_times_each_trans_in_curr_buffer_sampled, \
                avg_times_each_pos_trans_in_curr_buffer_sampled, \
                avg_times_each_neg_trans_in_curr_buffer_sampled, \
                num_pos_trans, num_neg_trans, num_pos_trans/buffer_size, \
                avg_num_pos_trans_per_batch, avg_num_neg_trans_per_batch, \
                avg_num_unique_pos_trans_per_batch, avg_num_unique_neg_trans_per_batch

    def __len__(self):
        return len(self._memory)

    def get_status(self):
        return self._memory

    @property
    def capacity(self):
        return self._capacity

    def __getitem__(self, idx):
        return self._memory[idx]

    @property
    def sample_method(self):
        return "uniform"



def create_track_f(replay_buffer, T_replay_track):
    def replay_track_f(tracker, t):
        if t>0 and t % T_replay_track == 0:
            track_keys = [
                    "avg_times_in_buffer_per_tran",
                    "avg_times_sampled_per_tran", "avg_times_sampled_per_pos_tran",
                    "avg_times_sampled_per_neg_tran", "num_pos_trans", "num_neg_trans",
                    "frac_pos_trans", "avg_num_pos_trans_per_batch", "avg_num_neg_trans_per_batch",
                    "avg_num_unique_pos_trans_per_batch", 
                    "avg_num_unique_neg_trans_per_batch"
                    ]
            # "max_q", "min_q", "avg_q"
            track_vals = replay_buffer.get_stats()
            tracker.tracks(track_keys, t, track_vals)
    def prio_replay_track_f(tracker, t):
        if t>0 and t % T_replay_track == 0:
            track_keys = [
                    "avg_times_in_buffer_per_tran",
                    "avg_times_sampled_per_tran", "avg_times_sampled_per_pos_tran",
                    "avg_times_sampled_per_neg_tran", "num_pos_trans", "num_neg_trans",
                    "frac_pos_trans", "avg_pos_trans_err", "avg_neg_trans_err",
                    "avg_num_pos_trans_per_batch", "avg_num_neg_trans_per_batch",
                    "avg_num_unique_pos_trans_per_batch", 
                    "avg_num_unique_neg_trans_per_batch", 
                    ]
            # "max_q", "min_q", "avg_q"
            track_vals = replay_buffer.get_stats()
            tracker.tracks(track_keys, t, track_vals)
    if replay_buffer.sample_method == "td_err_prioritized":
        return prio_replay_track_f
    else:
        return replay_track_f











