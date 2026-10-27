"""Install pinned official CPU runtime and download the requested GGUFs."""
import concurrent.futures
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'qwen3b': {'name': 'Qwen/Qwen2.5-3B-Instruct-GGUF',
        'revision': '7dabda4d13d513e3e842b20f0d435c732f172cbe',
        'filenames': ['qwen2.5-3b-instruct-q4_k_m.gguf']},
    'qwen7b': {'name': 'Qwen/Qwen2.5-7B-Instruct-GGUF',
        'revision': 'bb5d59e06d9551d752d08b292a50eb208b07ab1f',
        'filenames': ['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
                      'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf']},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    cache = ROOT/'.cache'
    cache.mkdir(exist_ok=True)
    archive = cache/'llama-b11935.zip'
    url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11935/llama-b11935-bin-win-cpu-x64.zip'
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    digest = sha(archive)
    runtime_lock = ROOT/'runtime.lock.json'
    if runtime_lock.exists():
        assert json.loads(runtime_lock.read_text())['archive_sha256'] == digest
    else:
        runtime_lock.write_text(json.dumps({'release': 'llama.cpp b11935 CPU x64',
            'download_url': url, 'archive_sha256': digest,
            'server_relative_path': '.cache/llama-b11935/llama-server.exe'}, indent=2))
    with zipfile.ZipFile(archive) as z:
        z.extractall(cache/'llama-b11935')
    lock_path = ROOT/'models.lock.json'
    lock = json.loads(lock_path.read_text())
    jobs = []
    for key, spec in SPECS.items():
        info = HfApi().model_info(spec['name'], revision=spec['revision'], files_metadata=True)
        files = []
        for name in spec['filenames']:
            item = next(s for s in info.siblings if s.rfilename == name)
            files.append({'name': name, 'sha256': item.lfs.sha256, 'bytes': item.size})
        lock[key] = {'name': spec['name'], 'revision': spec['revision'], 'files': files,
                     'quantization': 'Q4_K_M', 'backend': 'llama.cpp'}
        jobs.extend((spec, item) for item in files)
    lock_path.write_text(json.dumps(lock, indent=2))

    def download(job):
        spec, item = job
        print('Downloading', item['name'], flush=True)
        path = hf_hub_download(spec['name'], item['name'], revision=spec['revision'],
                               cache_dir=str(cache/'huggingface/hub'))
        assert sha(path) == item['sha256'], f'Checksum mismatch: {path}'
        print('Verified', item['name'], flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, jobs))
    print('Runtime and both model checkpoints ready.', flush=True)


if __name__ == '__main__':
    main()
