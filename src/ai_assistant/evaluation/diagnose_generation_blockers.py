"""Targeted verifier/regeneration experiment for three final DEV generation blockers."""
from __future__ import annotations
import json
from pathlib import Path
from ..generation.citations import canonicalize_citations, parse_grounded_result, validate_citations
from ..generation.config import GenerationConfig
from ..generation.prompts import GROUNDED_RESULT_SCHEMA, grounded_system_prompt, grounded_user_prompt
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from ..llm.models import LLMRequest
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, _context_from_dict

SOURCE = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_grounded_v1_citation_facing_final_v1.jsonl"
OUTPUT = DEFAULT_OUTPUT / "generation_dev_generation_blocker_verifier_v1.json"
TARGETS = ("V2-M03", "V2-M01", "V2-M11")
SCHEMA = {"type":"object","properties":{"status":{"type":"string","enum":["ok","supported_but_abstained","incomplete_evidence_use"]},"missing_handles":{"type":"array","items":{"type":"string"}}},"required":["status","missing_handles"],"additionalProperties":False}

def run() -> dict:
    rows={r["query_id"]:r for r in (json.loads(x) for x in SOURCE.read_text(encoding="utf-8").splitlines() if x)}
    cache={r["query_id"]:r for r in json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))["results"]}
    provider=get_llm(LLMSettings.from_env()); config=GenerationConfig(); results=[]
    for query_id in TARGETS:
        row,pack=rows[query_id],_context_from_dict(cache[query_id]["context_pack"])
        verifier=provider.generate(LLMRequest("Classify only; do not rewrite the answer. Return JSON.", f"QUESTION:\n{pack.original_query}\nANSWER:\n{row['generated_answer']}\nEVIDENCE:\n{pack.render_for_llm()}\nIs the answer unsupportedly abstained, or does it omit directly relevant evidence?", response_schema=SCHEMA,response_schema_name="completeness_verifier"))
        verdict=json.loads(verifier.text)
        regenerated=None
        if verdict["status"] != "ok":
            response=provider.generate(LLMRequest(grounded_system_prompt(config), grounded_user_prompt(pack)+f"\n\nVERIFIER RESULT: {json.dumps(verdict)}. Re-answer using all directly relevant supplied E-handles only.",response_schema=GROUNDED_RESULT_SCHEMA,response_schema_name="controlled_regeneration"))
            result,_=canonicalize_citations(parse_grounded_result(response.text),pack)
            try: validate_citations(result,pack); valid=True
            except Exception: valid=False
            regenerated={"answer":result.answer,"no_answer":result.no_answer,"citations":result.citations,"valid":valid}
        results.append({"query_id":query_id,"verifier":verdict,"regenerated":regenerated})
    report={"generation_performed":True,"targets":results}; OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); return report
if __name__ == "__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
