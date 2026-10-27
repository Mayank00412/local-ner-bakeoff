"""Local CPU llama.cpp adapter for the two larger Q4_K_M models."""
import atexit
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.common import PROMPT

ROOT = Path(__file__).resolve().parents[1]
BASE_MODELS = {
    'qwen3b': {'name': 'Qwen/Qwen2.5-3B-Instruct',
        'revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'parameters': 3085938688},
    'qwen7b': {'name': 'Qwen/Qwen2.5-7B-Instruct',
        'revision': 'a09a35458c702b33eeacc393d103063234e8bc28', 'parameters': 7615616512},
}


def chatml(text):
    return (f'<|im_start|>system\n{PROMPT}<|im_end|>\n'
            f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n')


def load_llama(key, revision, threads):
    from huggingface_hub import hf_hub_download
    spec = json.loads((ROOT/'models.lock.json').read_text())[key]
    if spec['revision'] != revision:
        raise ValueError('Revision does not match the downloaded model lock')
    files = [hf_hub_download(spec['name'], item['name'], revision=revision,
              cache_dir=str(ROOT/'.cache/huggingface/hub'), local_files_only=True)
             for item in spec['files']]
    runtime = json.loads((ROOT/'runtime.lock.json').read_text())
    server = ROOT/runtime['server_relative_path']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    log_path = ROOT/f'results/{key}_llama_server_{time.time_ns()}.log'
    log = log_path.open('wb')
    command = [str(server), '-m', files[0], '--host', '127.0.0.1', '--port', str(port),
               '-ngl', '0', '-c', '2048', '-np', '1', '-t', str(threads), '-tb', str(threads),
               '--no-warmup']
    process = subprocess.Popen(command, stdout=log, stderr=log,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def close():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        log.close()
    atexit.register(close)
    # Explicitly bypass proxy environment settings for loopback-only requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None, timeout=900):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base+path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as ex:
            if path == '/health':
                raise
            raise RuntimeError(f'llama.cpp {path}: HTTP {ex.code}: ' + ex.read().decode('utf-8', errors='replace')) from ex

    deadline = time.monotonic() + 600
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'llama-server exited; see {log_path}')
        try:
            request('/health', timeout=2)
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                close()
                raise RuntimeError('llama-server load timeout')
            time.sleep(.25)
    meta = request('/v1/models')['data'][0]['meta']

    def predict(text):
        response = request('/completion', {'prompt': chatml(text), 'n_predict': 384,
            'temperature': 0, 'seed': 0, 'stream': False, 'cache_prompt': False,
            'repeat_penalty': 1.1, 'repeat_last_n': 2048,
            'stop': ['<|im_end|>', '<|endoftext|>']})
        return None, response['content']

    return predict, {'name': spec['name'], 'revision': revision,
        'parameters': BASE_MODELS[key]['parameters'], 'stored_tensor_parameters': meta['n_params'],
        'base_model': BASE_MODELS[key], 'dtype': 'mixed GGUF Q4_K_M',
        'quantization': 'Q4_K_M', 'device': 'cpu', 'runtime': runtime['release'],
        'runtime_sha256': runtime['archive_sha256'], 'context_tokens': 2048,
        'repeat_penalty': 1.1, 'repeat_last_n': 2048, 'cache_prompt': False,
        'phase': 'post-review extension; fixed prompt/cleaner; previously inspected test set',
        'model_files': spec['files'], 'server_log': log_path.relative_to(ROOT).as_posix()}
