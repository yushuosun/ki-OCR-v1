"""Temporary CPU import guard: Python socket/HF requests and token lookups.

No persistent security/network settings change. Native syscall telemetry and GPU
model loading are outside this guard's scope.
"""
import importlib
import socket
from contextlib import ExitStack
from unittest.mock import patch

class OfflineImportGuard:
    def __init__(self):
        self.attempts = {'network':0, 'model_info_or_download':0, 'token_lookup':0}
        self.stack = ExitStack()

    def denied(self, family):
        def stop(*args, **kwargs):
            self.attempts[family] += 1
            raise RuntimeError('OFFLINE_CPU_IMPORT_FORBIDDEN_'+family.upper())
        return stop

    def install(self):
        for obj, names in ((socket.socket, ('connect','connect_ex')),
                           (socket, ('create_connection','getaddrinfo'))):
            for name in names:
                self.stack.enter_context(patch.object(obj, name, self.denied('network')))
        hf = importlib.import_module('huggingface_hub')
        self.stack.enter_context(patch.object(hf.HfApi, 'model_info', self.denied('model_info_or_download')))
        self.stack.enter_context(patch.object(hf, 'model_info', self.denied('model_info_or_download')))
        api = importlib.import_module('huggingface_hub.hf_api')
        if hasattr(api, 'model_info'):
            self.stack.enter_context(patch.object(api, 'model_info', self.denied('model_info_or_download')))
        self.stack.enter_context(patch.object(hf, 'hf_hub_download', self.denied('model_info_or_download')))
        self.stack.enter_context(patch.object(hf, 'snapshot_download', self.denied('model_info_or_download')))
        for module_name in ('huggingface_hub', 'huggingface_hub.utils',
                            'huggingface_hub.utils._auth', 'huggingface_hub.utils._headers',
                            'huggingface_hub.file_download'):
            module = importlib.import_module(module_name)
            for name in ('get_token', '_get_token_from_file', '_get_token_from_environment', '_get_token_from_google_colab'):
                if hasattr(module, name):
                    self.stack.enter_context(patch.object(module, name, self.denied('token_lookup')))
        return self

    def report(self):
        return {'attempts':dict(self.attempts),
                'scope':'Python socket/HF API calls during this CPU import process; native syscalls and GPU inference not measured'}

    def require_no_attempts(self):
        if any(self.attempts.values()):
            raise ValueError('Offline CPU import attempted forbidden calls: '+str(self.attempts))

    def close(self):
        self.stack.close()
