"""Small, predeclared official-test probe. Exact HTTP ranges, hard byte caps, no slices download."""
import argparse
import io
import json
import time
from pathlib import Path
import h5py
import httpx
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_url
from common import core, sha256
from main import _quality_metrics

# Fixed windows. Radiative files contain one test trajectory, so use two
# predetermined parameter files per system, rather than a nonexistent index 1.
SOURCES = [
    ("turbulent_radiative_layer_2D", "2ee7756575ff6f90981d0308cbcb6a2ab5995bdc", "turbulent_radiative_layer_tcool_0.03.hdf5", 40, "density"),
    ("turbulent_radiative_layer_2D", "2ee7756575ff6f90981d0308cbcb6a2ab5995bdc", "turbulent_radiative_layer_tcool_0.06.hdf5", 40, "density"),
    ("rayleigh_benard", "10e143ebedae8a9b1c699b95d5b1eb8feaae09b5", "rayleigh_benard_Rayleigh_1e8_Prandtl_1.hdf5", 100, "buoyancy"),
    ("rayleigh_benard", "10e143ebedae8a9b1c699b95d5b1eb8feaae09b5", "rayleigh_benard_Rayleigh_1e8_Prandtl_10.hdf5", 100, "buoyancy"),
    ("shear_flow", "fc867f856f306905cf94c1f5df978cc518a2048c", "shear_flow_Reynolds_1e5_Schmidt_1e0.hdf5", 60, "tracer"),
    ("shear_flow", "fc867f856f306905cf94c1f5df978cc518a2048c", "shear_flow_Reynolds_1e5_Schmidt_1e-1.hdf5", 60, "tracer"),
]

class BoundedRanges(io.RawIOBase):
    def __init__(self, url, size, budget=96*1024*1024):
        self.url=url; self.size=size; self.budget=budget; self.position=0; self.cache={}; self.downloaded=0
        self.client=httpx.Client(follow_redirects=True,timeout=90)
        self.block_size=256*1024
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0: raise ValueError("Negative offset")
        return self.position
    def read(self, count=-1):
        if count<0: raise ValueError("Unbounded reads are forbidden")
        end=min(self.size,self.position+count); chunks=[]
        while self.position<end:
            index=self.position//self.block_size
            if index not in self.cache:
                start=index*self.block_size; stop=min(self.size,start+self.block_size)-1
                expected=stop-start+1
                if self.downloaded+expected>self.budget: raise ValueError("Source byte budget exceeded")
                with self.client.stream("GET",self.url,headers={"Range":f"bytes={start}-{stop}","Accept-Encoding":"identity"}) as response:
                    if response.status_code!=206: raise ValueError("Server did not honor exact range; refusing full-file transfer")
                    if not response.headers.get("content-range","").startswith(f"bytes {start}-{stop}/"): raise ValueError("Unexpected content range")
                    data=response.read()
                    if len(data)!=expected: raise ValueError("Unexpected range length")
                self.cache[index]=data; self.downloaded+=len(data)
            offset=self.position%self.block_size; take=min(end-self.position,len(self.cache[index])-offset)
            chunks.append(self.cache[index][offset:offset+take]); self.position+=take
        return b"".join(chunks)
    def readinto(self, buffer):
        data=self.read(len(buffer)); buffer[:len(data)]=data; return len(data)
    def close(self):
        self.client.close(); super().close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--notebook",required=True)
    p.add_argument("--api",default="https://krishmalik-delta-v.hf.space")
    args=p.parse_args(); root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    api=HfApi(); client=httpx.Client(timeout=180)
    artifacts=api.dataset_info("KrishMalik/deltav-fluid",files_metadata=True)
    checkpoint=next(s for s in artifacts.siblings if s.rfilename=="checkpoints/domain_fluid.pt")
    report={"kind":"small official-test-split probe", "notebook_sha256":sha256(args.notebook),"artifact_revision":artifacts.sha,"checkpoint_lfs":checkpoint.lfs,"selection":"Predeclared source files; trajectories 0 and 1; one five-frame window each; no tuning on results", "scope":"Official test files are outside the train-only source path in the supplied notebook. Checkpoint-to-notebook lineage is supplied by the project owner, not cryptographically established. This small probe is not a comprehensive benchmark; the original random-window validation remains unchanged.","results":[]}
    report["selection"] = "Two fixed official test files per system; trajectory 0; one five-frame window each. Second-file selection replaced unavailable trajectory 1 after inspecting file dimensions, not prediction scores. No training or tuning on these results."
    previous = json.loads((root/"results.json").read_text()) if (root/"results.json").exists() else {"results":[]}
    for number,(system,revision,filename,start,scalar) in enumerate(SOURCES):
        path="data/test/"+filename; repo="polymathic-ai/"+system
        done=next((r for r in previous["results"] if r["system"]==system and r["source_file"]==path and r["source_revision"]==revision and r["start_index"]==start),None)
        if done and previous["artifact_revision"]==artifacts.sha and previous["notebook_sha256"]==report["notebook_sha256"] and (root/(done["id"]+".npy")).exists() and sha256(root/(done["id"]+".npy"))==done["window_sha256"]:
            report["results"].append(done); print("REUSE",done["id"],flush=True); continue
        info=api.get_paths_info(repo,[path],repo_type="dataset",revision=revision)[0]
        raw=BoundedRanges(hf_hub_url(repo,path,repo_type="dataset",revision=revision),info.size)
        with raw, h5py.File(raw,"r") as f:
            for trajectory in (0,):
                sample_id=f"test-{system}-{number%2}"
                local=root/(sample_id+".npy")
                scalar_name=scalar if scalar in f["t0_fields"] else "buoyancy"
                primary=f["t0_fields/"+scalar_name][trajectory,start:start+5]
                pressure=f["t0_fields/pressure"][trajectory,start:start+5]
                velocity=f["t1_fields/velocity"][trajectory,start:start+5]
                arr=np.stack([primary,pressure,velocity[...,0],velocity[...,1]],axis=-1).astype(np.float32)
                np.save(local,arr,allow_pickle=False)
                buf=io.BytesIO(); np.save(buf,arr[:4],allow_pickle=False)
                started=time.perf_counter()
                response=client.post(args.api+"/predict/field",data={"system":system},files={"file":("window.npy",buf.getvalue(),"application/octet-stream")})
                response.raise_for_status(); prediction=np.load(io.BytesIO(response.content),allow_pickle=False)
                truth,_=core.build_canonical(arr[4],system); truth=core.resample(torch.from_numpy(truth)).numpy()
                record={"id":sample_id,"system":system,"source_repo":repo,"source_revision":revision,"source_file":path,"trajectory":trajectory,"start_index":start,"window_sha256":sha256(local),"epoch":response.headers.get("x-checkpoint-epoch"),"request_seconds":round(time.perf_counter()-started,3),"quality":_quality_metrics(prediction,truth)}
                report["results"].append(record)
                (root/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False))
                print("MEASURED",sample_id,record["quality"],flush=True)
        print("RANGE_BYTES",system,raw.downloaded,flush=True)
    if api.dataset_info("KrishMalik/deltav-fluid").sha!=artifacts.sha: raise ValueError("Artifact revision changed during evaluation")
    print("COMPLETE",root/"results.json")

if __name__=="__main__": main()
