"""DEV-only validation of D20+B10 reranking and context top-15."""
from __future__ import annotations
import json
from pathlib import Path
from ..context.builder import ContextBuilder
from ..context.config import ContextConfig
from ..reranking.config import RerankerConfig
from ..reranking.reranked_retriever import RerankedRetriever
from .evaluate_context import _candidate, evaluate_configuration
from .evaluate_reranker import load_split_cases
from .evaluate_retrieval import DEFAULT_BUNDLE, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META

ROOT=Path(__file__).resolve().parents[3]
CACHE=ROOT/'artifacts/ai_assistant/cache/reranker_dev_d20_b10_cuda_v1.json'
JSONL_CACHE=ROOT/'artifacts/ai_assistant/cache/reranker_dev_d20_b10_cuda_v1.jsonl'
OUT=ROOT/'artifacts/ai_assistant/evaluation/upstream_candidate_d20_b10_cuda_v1.json'

def _rank(rows,cases):
 def at(k): return sum(bool(set(r['reranked_results'][:k]) & set()) for r in [])
 metrics={}
 for k in (1,3,5,10,15,20):
  recalls=[];hits=[]
  for c,r in zip(cases,rows):
   found={x['chunk_id'] for x in r['reranked_results'][:k]}; gold=set(c.relevant_chunk_ids);recalls.append(len(found&gold)/len(gold));hits.append(bool(found&gold))
  metrics[f'recall_at_{k}']=sum(recalls)/len(recalls);metrics[f'hit_at_{k}']=sum(hits)/len(hits)
 mrr=[];ndcg=[]
 for c,r in zip(cases,rows):
  gold=set(c.relevant_chunk_ids); ranks=[i+1 for i,x in enumerate(r['reranked_results'][:10]) if x['chunk_id'] in gold];mrr.append(1/min(ranks) if ranks else 0);ndcg.append(sum(1/__import__('math').log2(i+2) for i,x in enumerate(r['reranked_results'][:10]) if x['chunk_id'] in gold)/sum(1/__import__('math').log2(i+2) for i in range(min(10,len(gold)))))
 return {**metrics,'mrr_at_10':sum(mrr)/len(mrr),'ndcg_at_10':sum(ndcg)/len(ndcg)}
def run():
 cases,_=load_split_cases(DEFAULT_V2_DATASET,DEFAULT_V2_META,DEFAULT_SPLITS,DEFAULT_BUNDLE,'dev')
 config=RerankerConfig(device='cuda',candidate_strategy='union_d20_b10',dense_candidate_k=20,bm25_candidate_k=10,rerank_candidate_limit=30,output_top_k=30)
 rows=[];reranker=RerankedRetriever(config)
 runtime=reranker.runtime_info()
 if runtime['resolved_device'] != 'cuda': raise RuntimeError('CUDA reranker requirement was not met')
 for c in cases:
  result=reranker.retrieve(c.query);row={'query_id':c.query_id,'query':c.query,'reranked_results':[x.to_dict() for x in result.reranked_results]};rows.append(row);CACHE.parent.mkdir(parents=True,exist_ok=True);CACHE.write_text(json.dumps({'identity':{'corpus_manifest':json.loads((DEFAULT_BUNDLE/'release_manifest.json').read_text()),'dense_k':20,'bm25_k':10,'reranker_model':config.model_id,'reranker_revision':config.model_revision,'context_version':'chunk_only_top15_2000','runtime':runtime},'results':rows},ensure_ascii=False,indent=2),encoding='utf-8');JSONL_CACHE.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
 old={x['query_id']:x for x in json.loads((ROOT/'artifacts/ai_assistant/cache/context_eval_reranked_v1.json').read_text(encoding='utf-8'))['results']};old_rows=[old[c.query_id] for c in cases]
 report={'configuration':config.to_dict(),'runtime':runtime,'old_reranker':_rank(old_rows,cases),'new_reranker':_rank(rows,cases),'context':{str(k):evaluate_configuration(cases,rows,ContextConfig(candidate_top_k=k,max_context_tokens=2000)) for k in (10,15)},'cache':str(CACHE),'jsonl_cache':str(JSONL_CACHE)}
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return report
if __name__=='__main__': print(json.dumps(run(),ensure_ascii=False,indent=2))
