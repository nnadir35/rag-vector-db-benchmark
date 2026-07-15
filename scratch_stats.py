import json
import numpy as np
from scipy import stats

self_data = json.load(open("experiments/results/20260715_190239_baseline_ollama_squad.json"))
sep_data = json.load(open("experiments/results/20260715_190607_baseline_ollama_separate_judge.json"))

# Query metnine göre eşleştir (sıra garantisi olmayabilir diye güvenli yöntem)
sep_by_query = {r["query"]: r["metrics"] for r in sep_data["results"]}

faith_self, faith_sep = [], []
rel_self, rel_sep = [], []
unmatched = 0

for r in self_data["results"]:
    q = r["query"]
    if q not in sep_by_query:
        unmatched += 1
        continue
    faith_self.append(r["metrics"]["faithfulness"])
    faith_sep.append(sep_by_query[q]["faithfulness"])
    rel_self.append(r["metrics"]["relevancy"])
    rel_sep.append(sep_by_query[q]["relevancy"])

print(f"Eşleşen sorgu sayısı: {len(faith_self)}  |  Eşleşmeyen: {unmatched}")

faith_self, faith_sep = np.array(faith_self), np.array(faith_sep)
rel_self, rel_sep = np.array(rel_self), np.array(rel_sep)

def analyze(name, a, b):
    diff = a - b
    print(f"\n--- {name} ---")
    print(f"Ortalama fark (self - separate): {diff.mean():.4f}  (std: {diff.std():.4f})")

    # Normallik testi (hangi testi kullanmamız gerektiğine karar vermek için)
    shapiro_stat, shapiro_p = stats.shapiro(diff)
    print(f"Shapiro-Wilk normallik testi: W={shapiro_stat:.4f}, p={shapiro_p:.4f}"
          f"  -> {'Normal DEĞİL' if shapiro_p < 0.05 else 'Normal'} dağılım")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(a, b)
    print(f"Paired t-test: t({len(a)-1})={t_stat:.4f}, p={t_p:.6f}"
          f"  -> {'ANLAMLI' if t_p < 0.05 else 'anlamlı değil'} (α=0.05)")

    # Wilcoxon signed-rank test (normal dağılım varsayımı gerektirmez)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
        print(f"Wilcoxon signed-rank: W={w_stat:.4f}, p={w_p:.6f}"
              f"  -> {'ANLAMLI' if w_p < 0.05 else 'anlamlı değil'} (α=0.05)")
    except ValueError as e:
        print(f"Wilcoxon test çalıştırılamadı: {e}")

    # Etki büyüklüğü: paired Cohen's d
    cohens_d = diff.mean() / diff.std(ddof=1)
    print(f"Cohen's d (paired): {cohens_d:.4f}"
          f"  -> {'küçük' if abs(cohens_d)<0.2 else 'orta' if abs(cohens_d)<0.5 else 'büyük' if abs(cohens_d)<0.8 else 'çok büyük'} etki")

analyze("Faithfulness", faith_self, faith_sep)
analyze("Relevancy", rel_self, rel_sep)

# En çok farklılaşan 3 örneği bul (tez metninde nitel örnek olarak kullanılabilir)
diffs = [(r["query"], r["metrics"]["faithfulness"], sep_by_query[r["query"]]["faithfulness"])
         for r in self_data["results"] if r["query"] in sep_by_query]
diffs.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
print("\n--- En çok farklılaşan 3 sorgu (faithfulness) ---")
for q, sf, pf in diffs[:3]:
    print(f"Soru: {q[:80]}\n  self={sf:.2f}  separate={pf:.2f}  fark={sf-pf:+.2f}")
