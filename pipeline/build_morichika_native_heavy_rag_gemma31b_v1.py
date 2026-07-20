"""Build a clean MORICHIKA-native Gemma-4-31B Kaggle notebook.

The emitted notebook is authored from MORICHIKA runtime components only.  It
contains no imported notebook cells, Phase-1 labels, or legacy QA banks.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline.build_morichika_final31b_integrated_v2 import (
    LOADER, OVERRIDES, RETRIEVAL, RUN, SETUP, code, markdown
)

OUT = ROOT / "artifacts/kaggle/morichika_heavy_rag_gemma31b_v1_20260720"
PACKAGE = ROOT / "artifacts/kaggle/morichika_phase2_retrieval_strict_v3_20260720"
CONTEXT = ROOT / "pipeline/contextual_policy_v4_runtime.py"


MODEL_RUNTIME = r'''# MORICHIKA-owned, hash-verified offline llama.cpp runtime.
runtime_roots=[p.parent for p in INPUT_ROOT.rglob("runtime_manifest.json") if "morichika-offline-runtime" in str(p).lower()]
if len(runtime_roots)!=1: raise RuntimeError(f"expected one MORICHIKA runtime, got {runtime_roots}")
runtime_root=runtime_roots[0]
runtime_manifest=json.loads((runtime_root/"runtime_manifest.json").read_text(encoding="utf-8-sig"))
if runtime_manifest.get("dataset_id")!="ishtyy/morichika-offline-runtime-20260720": raise RuntimeError("wrong runtime dataset")
for spec in runtime_manifest["files"]:
    path=runtime_root/spec["path"]
    if not path.is_file() or path.stat().st_size!=int(spec["bytes"]) or file_sha256(path)!=spec["sha256"]:
        raise RuntimeError(f"runtime hash gate failed: {path}")
wheel=next(runtime_root.glob("llama_cpp_python-*.whl"))
subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-deps","--force-reinstall",str(wheel)],check=True)
from llama_cpp import Llama, LlamaGrammar

def compact_field(value, limit=None):
    text=str(value or "").strip()
    return text if limit is None or len(text)<=limit else text[:limit]

def compact_context(value,max_chars=6000):
    text=str(value or "").strip()
    if len(text)<=max_chars: return text
    half=max_chars//2
    return text[:half]+"\n[...MIDDLE OMITTED ONLY AFTER SOURCE-LINKED FULL-COVERAGE NOTE EXTRACTION...]\n"+text[-half:]

def locate_model():
    candidates=[]
    for p in INPUT_ROOT.rglob("*.gguf"):
        name=p.name.casefold()
        if "31b" in name and ("q4_0" in name or "q4" in name): candidates.append(p)
    if len(candidates)!=1: raise RuntimeError(f"expected one official Gemma31B Q4 GGUF, got {candidates}")
    return candidates[0]

AB_GRAMMAR=LlamaGrammar.from_string('root ::= "A" | "B"')
def load_judge():
    model_path=locate_model()
    print("MORICHIKA model",model_path)
    return Llama(model_path=str(model_path),n_ctx=Q4_N_CTX,n_batch=Q4_N_BATCH,n_ubatch=Q4_N_UBATCH,
                 n_gpu_layers=-1,tensor_split=[0.5,0.5],main_gpu=0,flash_attn=True,
                 offload_kqv=True,logits_all=False,seed=SEED,verbose=False)

def cleanup_judge(judge):
    try: judge.close()
    except Exception: pass

def render_chat(system,user):
    return "<bos><start_of_turn>user\n"+system+"\n\n"+user+"<end_of_turn>\n<start_of_turn>model\n"

def one_letter(judge,prompt):
    # Reserve room for the verdict and truncate only from the left of the
    # rendered prompt; model-facing evidence is already ranked and bounded.
    tokens=judge.tokenize(prompt.encode("utf-8"),add_bos=False,special=True)
    if len(tokens)>=Q4_N_CTX-8:
        tokens=tokens[-(Q4_N_CTX-8):]
        prompt=judge.detokenize(tokens).decode("utf-8",errors="ignore")
    out=judge.create_completion(prompt=prompt,max_tokens=1,temperature=0.0,top_p=1.0,
                                repeat_penalty=1.0,grammar=AB_GRAMMAR,seed=SEED)
    letter=str(out["choices"][0]["text"]).strip().upper()[:1]
    if letter not in {"A","B"}: raise RuntimeError(f"constrained verdict failed: {out}")
    return letter

def score_rows(frame,cache_name,judge):
    cache=WORK_DIR/cache_name
    done={}
    if cache.is_file():
        prior=pd.read_csv(cache)
        done={str(r.row_key):r._asdict() for r in prior.itertuples(index=False)}
    records=[]
    for index,row in frame.iterrows():
        key=str(row.row_key); ph=_q4_prompt_hash(row)
        prior=done.get(key)
        if prior and str(prior.get("prompt_hash"))==ph:
            records.append(prior); continue
        normal=one_letter(judge,render_chat(SYSTEM_PROMPT,build_user_prompt(row,False)))
        reverse=one_letter(judge,render_chat(SYSTEM_PROMPT,build_user_prompt(row,True)))
        p_normal=1.0 if normal=="A" else 0.0
        p_reverse=1.0 if reverse=="B" else 0.0
        if p_normal==p_reverse:
            p_faithful=p_normal
        else:
            tie_user=build_user_prompt(row,False)+"\nThe two order-balanced passes disagreed. Re-check counterevidence, exact slot and lane boundary. Emit A or B."
            tie=one_letter(judge,render_chat(SYSTEM_PROMPT,tie_user))
            p_faithful=1.0 if tie=="A" else 0.0
        rec={"row_key":key,"prompt_hash":ph,"p_normal":p_normal,"p_reverse":p_reverse,
             "p_faithful":p_faithful,"order_gap":abs(p_normal-p_reverse),
             "normal_letter":normal,"reverse_letter":reverse}
        records.append(rec)
        if len(records)%CHECKPOINT_EVERY==0:
            pd.DataFrame(records).to_csv(cache,index=False)
            print("checkpoint",len(records),"/",len(frame))
    result=pd.DataFrame(records)
    result.to_csv(cache,index=False)
    return result
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    manifest=json.loads((PACKAGE/"bundle_manifest.json").read_text(encoding="utf-8-sig"))
    loader=LOADER.replace("__MANIFEST_ID__",manifest["manifest_id"]).replace("__PACKAGE_ID__",manifest["package_id"])
    setup=SETUP.replace('MODEL_ID = "google/gemma-4-31B-it"','MODEL_ID = "google/gemma-4-31B-it"')
    retrieval=RETRIEVAL.replace(
        '("spelling_grammar_rule",r"শুদ্ধ\\s*বানান|ব্যাকরণ(?:ের)?\\s*নিয়ম|ব্যাকরণ(?:ের)?\\s*নিয়ম"),',
        '("spelling_grammar_rule",r"শুদ্ধ\\s*বানান|ব্যাকরণ(?:ের)?\\s*নিয়ম|ব্যাকরণ(?:ের)?\\s*নিয়ম"),\n'
        '        ("definition_theory_rule",r"সংজ্ঞা|তত্ত্ব|সূত্র|বিধান|কাকে\\s*বলে|কি\\s*বোঝা[য়য়]|কী\\s*বোঝা[য়য়]"),'
    )
    overrides=OVERRIDES.replace("morichika-final31b-integrated-v2-17x26x15","morichika-heavy-rag-gemma31b-v1-17x26x15").replace(
        "then other corroboration", "then Wikipedia/other corroboration last"
    )
    run=RUN.replace("morichika_final31b_integrated_v2_scores.csv","morichika_heavy_rag_gemma31b_v1_scores.csv").replace('"version":"morichika-final31b-integrated-v2"','"version":"morichika-heavy-rag-gemma31b-v1"')
    nb={"cells":[
        markdown("# MORICHIKA Heavy RAG Gemma31B v1\n\nNative strict-v3 retrieval, context-policy-v4 and Gemma31B dual-order verification. No legacy notebook/data-dump logic."),
        code(setup),code(loader),code(CONTEXT.read_text(encoding="utf-8")),code(retrieval),
        code(MODEL_RUNTIME),code(overrides),code(run)],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},
        "nbformat":4,"nbformat_minor":5}
    OUT.mkdir(parents=True,exist_ok=True)
    nb_path=OUT/"morichika-heavy-rag-gemma31b-v1.ipynb"
    nb_path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8",newline="\n")
    meta={"id":"ishtyy/morichika-heavy-rag-gemma31b-v1-20260720","title":"MORICHIKA Heavy RAG Gemma31B v1 20260720",
          "code_file":nb_path.name,"language":"python","kernel_type":"notebook","is_private":True,
          "enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":["morichika","bengali","offline","heavy-rag"],
          "dataset_sources":["ishtyy/morichika-phase2-retrieval-strict-v3-20260720","ishtyy/morichika-offline-runtime-20260720"],
          "kernel_sources":[],"competition_sources":["bengali-hallucination"],
          "model_sources":["google/gemma-4/Gguf/gemma-4-31b-it-qat-q4_0-gguf/2"],"machine_shape":"NvidiaTeslaT4"}
    meta_path=OUT/"kernel-metadata.json"; meta_path.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    receipt={"notebook":str(nb_path),"notebook_sha256":sha(nb_path),"metadata_sha256":sha(meta_path),
             "retrieval_manifest_id":manifest["manifest_id"],"retrieval_package_id":manifest["package_id"],
             "context_policy_sha256":sha(CONTEXT),"native_notebook":True,"no_legacy_data_dump":True,"no_submit":True}
    (OUT/"MORICHIKA_NATIVE_BUILD_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    return receipt


if __name__=="__main__": print(json.dumps(build(),indent=2))
