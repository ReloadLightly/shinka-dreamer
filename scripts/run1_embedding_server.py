"""CPU-only genuine ONNX embeddings through native Shinka's local backend.

General English BGE encoder applied to code, not a code-trained semantic oracle.
Native Shinka supplies at most 10,000 source characters. Split those into <=480
model tokens using source offsets; average normalized chunk vectors and normalize.
All inference is local; startup requires already downloaded hash-bound weights.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
NAME = 'bge-small-code-chunks-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--results', default='results/campaign-v4-run1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    directory = ROOT / '.runtime/run1-embedding/model'
    output = ROOT / args.results
    output.mkdir(parents=True, exist_ok=True)
    identity = json.loads((directory.parent/'download.json').read_text())
    identity.update({'model': 'BAAI/bge-small-en-v1.5', 'endpoint_model': NAME,
                     'dim':384, 'threads':1, 'device':'CPUExecutionProvider',
                     'aggregation':'source-offset chunks <=480 wordpiece tokens; unweighted mean, L2 normalize',
                     'limitation':'general English encoder on code; upstream first10000characters',
                     'packages':{p:importlib.metadata.version(p) for p in ('fastembed','onnxruntime','tokenizers','numpy')},
                     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     'file_sha256':{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in directory.rglob('*') if p.is_file() and '.cache' not in str(p)}})
    identity_path=output/'embedding-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Embedding identity drift; use a new treatment')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    encoder = TextEmbedding('BAAI/bge-small-en-v1.5', specific_model_path=str(directory),
                            threads=1, providers=['CPUExecutionProvider'], local_files_only=True)
    splitter = Tokenizer.from_file(str(directory/'tokenizer.json'))
    splitter.no_truncation(); splitter.no_padding()

    def embed(source):
        enc = splitter.encode(source, add_special_tokens=False)
        chunks = [source[enc.offsets[i][0]:enc.offsets[min(i+479,len(enc.offsets)-1)][1]]
                  for i in range(0,len(enc.offsets),480)] or ['']
        vectors = np.asarray(list(encoder.embed(chunks,batch_size=4)),dtype=np.float64)
        v = vectors.mean(axis=0); v /= np.linalg.norm(v)
        if v.shape != (384,) or not np.isfinite(v).all():
            raise ValueError('Malformed local embedding')
        return v.tolist(),len(enc.ids),len(chunks)

    fixture=['def square(x):\n    return x*x\n','def square(x):\n    return x*x\n',
             'def walk(memory, observation):\n    memory["seen"].append(observation)\n    return memory\n']
    begun=time.monotonic(); cpu=time.process_time()
    vectors=[embed(s)[0] for s in fixture]
    check={'identical_exact':vectors[0]==vectors[1],
           'different_cosine':float(np.dot(vectors[0],vectors[2])),
           'dimensions':len(vectors[0]),'wall_seconds':time.monotonic()-begun,
           'cpu_seconds':time.process_time()-cpu,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'kind':'local deterministic fixture, not evolutionary evidence'}
    if not check['identical_exact'] or check['different_cosine']>=.9999:
        raise ValueError('Embedding fixture failed')
    (output/'embedding-fixture.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({'ready':True, 'bind':'127.0.0.1','port':args.port, **check}),flush=True)
    if args.check_only:
        return

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_GET(self):
            data={'ready':True,'model':NAME} if self.path=='/health' else {'error':'not found'}
            self.send_response(200 if self.path=='/health' else 404)
            self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        def do_POST(self):
            if self.path != '/v1/embeddings':
                self.send_error(404);return
            started=time.monotonic(); cpu=time.process_time()
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2*1024**2: raise ValueError('Input too large')
                req=json.loads(self.rfile.read(size))
                if req.get('model')!=NAME: raise ValueError('Unknown model')
                texts=req['input']
                if isinstance(texts,str):texts=[texts]
                if not isinstance(texts,list) or not 1<=len(texts)<=64 or not all(isinstance(t,str) for t in texts):
                    raise ValueError('Expected text inputs')
                rows=[];tokens=0;chunks=0
                for i,t in enumerate(texts):
                    vector,n,c=embed(t);tokens+=n;chunks+=c
                    rows.append({'object':'embedding','index':i,'embedding':vector})
                payload={'object':'list','data':rows,'model':NAME,
                         'usage':{'prompt_tokens':tokens,'total_tokens':tokens}}
                event={'utc':datetime.now(timezone.utc).isoformat(),'status':'ok','inputs':len(texts),
                       'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in texts],
                       'tokens':tokens,'chunks':chunks,'wall_seconds':time.monotonic()-started,
                       'cpu_seconds':time.process_time()-cpu,
                       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                with (output/'embedding-calls.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
            except Exception as exc:
                with (output/'embedding-calls.jsonl').open('a') as f:f.write(json.dumps({'status':'error','error':str(exc)})+'\n')
                self.send_error(400,str(exc))
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':
    main()
