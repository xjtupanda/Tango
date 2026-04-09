# ------------------------------------------------------------------------
# Copyright 2026 Shukang Yin
# Timer utils for profiling.
# ------------------------------------------------------------------------


from transformers import LogitsProcessor
import contextlib
import torch


class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self, start_event):
        self.start_event = start_event
        self.ttft_event = torch.cuda.Event(enable_timing=True)
        self.ttft_ms = 0
        self.is_first_token = True
        self.token_count = 0

    def __call__(self, input_ids, scores):
        if self.is_first_token:
            self.ttft_event.record()
            torch.cuda.synchronize()
            self.ttft_ms = self.start_event.elapsed_time(self.ttft_event)
            self.is_first_token = False
            
        self.token_count += 1
        return scores


class ModuleProfiler(contextlib.ContextDecorator):
    def __init__(self, module):
        self.module = module
        self.time_ms = 0
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.hook_handles = []

    def __enter__(self):
        def pre_hook(mod, inp):
            self.start_event.record()
        def post_hook(mod, inp, out):
            self.end_event.record()
            torch.cuda.synchronize()
            self.time_ms += self.start_event.elapsed_time(self.end_event)
            
        if self.module is not None:
            self.hook_handles.append(self.module.register_forward_pre_hook(pre_hook))
            self.hook_handles.append(self.module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.hook_handles:
            handle.remove()