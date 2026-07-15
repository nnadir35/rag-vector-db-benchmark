import json, glob

self_file = sorted(glob.glob("experiments/results/*baseline_ollama_squad*.json"))[-1]
sep_file = sorted(glob.glob("experiments/results/*baseline_ollama_separate_judge*.json"))
sep_file = sep_file[-1] if sep_file else None

self_data = json.load(open(self_file))
print("Self-judge dosyası:", self_file)
print("Self-judge özet:", self_data.get("metrics_summary", {}))

if sep_file:
    sep_data = json.load(open(sep_file))
    print()
    print("Separate-judge dosyası:", sep_file)
    print("Separate-judge özet:", sep_data.get("metrics_summary", {}))

    print()
    print("=== FARK (faithfulness / relevancy) ===")
    sf = self_data["metrics_summary"].get("faithfulness")
    sr = self_data["metrics_summary"].get("answer_relevancy") or self_data["metrics_summary"].get("relevancy")
    pf = sep_data["metrics_summary"].get("faithfulness")
    pr = sep_data["metrics_summary"].get("answer_relevancy") or sep_data["metrics_summary"].get("relevancy")
    print(f"faithfulness: self={sf}  separate={pf}  fark={round(sf-pf,4) if sf and pf else 'N/A'}")
    print(f"relevancy:    self={sr}  separate={pr}  fark={round(sr-pr,4) if sr and pr else 'N/A'}")

    print("\n=== EN BÜYÜK 3 FARK (FAITHFULNESS) ===")
    self_queries = {q["query_id"]: q for q in self_data.get("per_query_results", [])}
    sep_queries = {q["query_id"]: q for q in sep_data.get("per_query_results", [])}
    
    diffs = []
    for q_id, q_self in self_queries.items():
        q_sep = sep_queries.get(q_id)
        if q_sep:
            f_self = q_self.get("metrics", {}).get("faithfulness")
            f_sep = q_sep.get("metrics", {}).get("faithfulness")
            if f_self is not None and f_sep is not None:
                diffs.append({
                    "query": q_self.get("query_text", q_id),
                    "self": f_self,
                    "sep": f_sep,
                    "diff": abs(f_self - f_sep)
                })
    
    diffs.sort(key=lambda x: x["diff"], reverse=True)
    for idx, d in enumerate(diffs[:3]):
        print(f"{idx+1}. Soru: {d['query']}\n   Self: {d['self']} | Separate: {d['sep']} | Fark: {d['diff']:.4f}\n")
else:
    print("Separate-judge sonuç dosyası bulunamadı.")
