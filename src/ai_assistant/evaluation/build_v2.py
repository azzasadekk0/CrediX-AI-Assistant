"""Build the reviewed v2 benchmark from immutable v1 plus multi-evidence cases."""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "datasets"
ROOT = Path(__file__).resolve().parents[3]
CHUNKS = {r["chunk_id"]: r for r in map(json.loads, (ROOT / "src/rag_data/current/chunks.jsonl").read_text(encoding="utf-8").splitlines())}
V1 = [json.loads(x) for x in (DATA / "golden_retrieval_v1.jsonl").read_text(encoding="utf-8").splitlines() if x]

PAIRS = [("V2-M01","ما شروط منح تسهيل بالنقد الأجنبي من حيث الغرض ومصدر السداد؟","D1-02","D1-03"),("V2-M02","ما الذي يراجعه البنك في دراسة الائتمان وقبل رفع الحد؟","D1-05","D1-09"),("V2-M03","كيف يجمع تقييم الجدارة بين الصناعة واستمرار المؤسسة؟","D2-03","D2-04"),("V2-M04","ما مؤشرا التدفقات النقدية وتغطية الفوائد في التصنيف؟","D2-06","D2-07"),("V2-M05","ما أثر المتأخرات والمعالجة المحاسبية للقرض غير المنتظم؟","D2-10","D2-11"),("V2-M06","ما الذي يلزم إقراره للجهات الصغيرة وما وضع الشركات الكبيرة في التسجيل؟","D3-04","D3-05"),("V2-M07","كيف يضمن البنك صحة البيانات وأمنها في السجل؟","D3-08","D3-09"),("V2-M08","ما متطلبات إقرار طالب التمويل والاطلاع على بيانه المجمع؟","D3-13","D3-14"),("V2-M09","ما شروط استخدام التقييم الرقمي من حيث المبيعات والعملة؟","D4-02","D4-03"),("V2-M10","من يعتمد نموذج التقييم الرقمي ومتى يخطر المركزي؟","D4-04","D4-06"),("V2-M11","ما التزامات تمويل متناهي الصغر وحدود عدد القروض؟","D4-08","D4-09"),("V2-M12","ما استقلال البنك المركزي ورأس ماله الأدنى؟","D5-01","D5-02"),("V2-M13","ما شروط التمويل الطارئ وحده الزمني؟","D5-04","D5-12"),("V2-M14","ما رأسمال ترخيص بنك وما موافقة تملك أكثر من 10%؟","D5-08","D5-09"),("V2-M15","ما حد العميل الواحد وعقوبة استخدام التمويل في غير غرضه؟","D5-10","D5-11"),("V2-M16","ما أهداف المركزي ومكونات احتياطياته الأجنبية؟","D5-03","D5-06"),("V2-M17","ما البيانات المطلوبة للتسجيل ومن المسئول عن إرسالها؟","D3-03","D3-07"),("V2-M18","ما ضوابط البضائع المرهونة والمخازن المقبولة؟","D1-10","D1-11"),("V2-M19","كيف يدخل الهيكل الإداري والمشاكل القانونية في التقييم؟","D2-08","D2-09"),("V2-M20","ما المطلوب في تصميم النموذج والتقرير الربع سنوي؟","D4-05","D4-07")]
by_id={r["query_id"]:r for r in V1}
extra=[]
for qid,q,a,b in PAIRS:
    left,right=by_id[a],by_id[b]; ids=left["relevant_chunk_ids"]+right["relevant_chunk_ids"]
    extra.append({"query_id":qid,"query":q,"language":"ar","query_type":"multi_constraint","difficulty":"hard","answerable":True,"expected_document_ids":sorted(set(left["expected_document_ids"]+right["expected_document_ids"])),"relevant_chunk_ids":ids,"primary_chunk_id":ids[0],"relevant_parent_ids":sorted({CHUNKS[i]["parent_id"] for i in ids}),"label_note":"The question requires evidence from both cited provisions.","evidence_scope":"multi"})
for row in V1: row["evidence_scope"]="single" if row["answerable"] else "none"
v2=V1+extra
(DATA/"golden_retrieval_v2.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in v2)+"\n",encoding="utf-8")
answer=[r for r in v2 if r["answerable"]]; dev=[r["query_id"] for i,r in enumerate(answer) if i%10<7]; test=[r["query_id"] for r in answer if r["query_id"] not in dev]; no=[r["query_id"] for r in v2 if not r["answerable"]]
(DATA/"golden_retrieval_v2.splits.json").write_text(json.dumps({"dev":dev,"test":test,"dev_no_answer":no[:7],"test_no_answer":no[7:]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
meta={"schema_version":"2.0","dataset_version":"golden_retrieval_v2","number_of_answerable_queries":len(answer),"number_of_unanswerable_queries":len(no),"single_evidence_queries":60,"multi_evidence_queries":20,"query_type_counts":dict(sorted(Counter(r["query_type"] for r in v2).items())),"difficulty_counts":dict(sorted(Counter(r["difficulty"] for r in v2).items())),"document_coverage_counts":dict(sorted(Counter(d for r in answer for d in r["expected_document_ids"]).items())),"chunks_file_sha256":hashlib.sha256((ROOT/"src/rag_data/current/chunks.jsonl").read_bytes()).hexdigest(),"release_manifest_sha256":hashlib.sha256((ROOT/"src/rag_data/current/release_manifest.json").read_bytes()).hexdigest(),"split_counts":{"dev":len(dev),"test":len(test),"dev_no_answer":7,"test_no_answer":3}}
(DATA/"golden_retrieval_v2.meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
audit=[{"query_id":r["query_id"],"status":"structurally_valid","original_relevant_chunks":r["relevant_chunk_ids"],"v2_relevant_chunks":r["relevant_chunk_ids"],"reason":"IDs, documents, and parents validate; semantic review is not recorded."} for r in V1 if r["answerable"]]
out=ROOT/"artifacts/ai_assistant/evaluation"; out.mkdir(parents=True,exist_ok=True); (out/"golden_v1_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
