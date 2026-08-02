# Tez Deney Planı v2 — RAG Vector DB Benchmark

Danışman kararı: **10K/50K ölçekte kalite farkı gösterilecek.**

v2 değişiklikleri: sabit sorgu seti confound'u eklendi, korpus seçenekleri
genişletildi (BEIR / MS MARCO / Türkçe), filtreli-hibrit arama deneyi eklendi,
multimodal yerine embedding boyutu ablation'ı kondu, deneyler fazlara ayrıldı.

---

# BÖLÜM 0 — Deneylerden ÖNCE yapılacak kod değişiklikleri

Bunlar yapılmadan koşulan hiçbir sonuç tezde kullanılamaz.

## 0.1 — Index tipi standardizasyonu (KRİTİK)

Mevcut durum:

| DB            | Index                            | Arama tipi  |
| ------------- | -------------------------------- | ----------- |
| FAISS         | `IndexFlatIP`                    | Exact       |
| Milvus        | `index_type="FLAT"`              | Exact       |
| Chroma        | HNSW (varsayılan)                | Approximate |
| Qdrant        | HNSW (varsayılan)                | Approximate |
| Weaviate      | HNSW (varsayılan)                | Approximate |
| ElasticSearch | HNSW, `num_candidates=100` sabit | Approximate |

Yapılacak:

1. **FAISS'i çift moda çıkar** — `FAISSRetrieverConfig`'e `index_type` alanı ekle:
   - `flat` (mevcut, exact) → **referans / ground truth**
   - `hnsw` (`faiss.IndexHNSWFlat`, M ve efConstruction parametreli)
2. **Milvus'u HNSW'ye çevir** — `milvus_retriever.py:145` `index_type="FLAT"` →
   `index_type="HNSW"`, `params={"M":16,"efConstruction":200}`. FLAT modu config ile korunsun.
3. **Ortak HNSW parametre seti** — altı DB'ye de aynısı: `M=16, ef_construction=200, ef_search=64`
   - Weaviate: `Configure.VectorIndex.hnsw(max_connections=16, ef_construction=200, ef=64)`
   - Qdrant: `HnswConfigDiff(m=16, ef_construct=200)` + aramada `hnsw_ef=64`
   - Chroma: `hnsw:M`, `hnsw:construction_ef`, `hnsw:search_ef`
   - ES: mapping'de `index_options: {type: hnsw, m: 16, ef_construction: 200}`
4. **ES `num_candidates`'ı config'e taşı** — `elasticsearch_retriever.py:263` hardcoded 100.
   ES'in ef_search karşılığı; sabit kalırsa büyük ölçekte ES haksız yere kaybeder.

## 0.2 — Sabit sorgu seti (KRİTİK — v2'de eklendi)

**Sorun:** `_select_queries_for_documents()` (`benchmark_db.py:279`) yalnızca gold
dokümanı indekslenmiş korpusta olan sorguları seçiyor. Sonuç: her ölçekte **farklı
sorgu seti** kullanılıyor. Ölçekler arası recall düşüşünün ne kadarı samanlığın
büyümesinden, ne kadarı sorgu setinin değişmesinden — ayrıştırılamıyor.

**Çözüm:** Dokümanlar prefix olarak yükleniyor (`val_docs[:num_documents]`), yani
küçük ölçeğin doküman kümesi büyük ölçeğinkinin alt kümesi. Bu iç içe geçmişlik
sayesinde:

> Sorgu setini **en küçük ölçekten** seç, sabitle, **her ölçekte aynısını** kullan.

100 doküman ≈ 990 sorgu barındırır → 500'lük sabit set rahat çıkar.
Böylece ölçekler arası recall düşüşü **saf distractor etkisi** olur.

İkincil olarak "ölçeğe özgü sorgu seti" sonuçları da raporlanabilir, ama ana
tabloda sabit set kullanılacak.

## 0.3 — Değişken sınıflandırması: neyi sabit tutmalı?

| Değişken         | Türü                                                            | Karar                                                       |
| ---------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| `top_k`          | **Treatment** — bulguyu doğrudan değiştirir                     | Tüm ölçeklerde **10 sabit**. Ayrı ablation olarak süpürülür |
| `num_queries`    | **Precision** — ortalamayı yanlı yapmaz, hata payını değiştirir | Tüm ölçeklerde **500 sabit** (std'ler karşılaştırılabilsin) |
| Tekrar sayısı    | **Noise** — varyans tahmini                                     | Tüm ölçeklerde **3 sabit**                                  |
| Chunk boyutu     | **Treatment**                                                   | Ana seride 512 sabit; ayrı ablation                         |
| Embedding modeli | **Treatment**                                                   | Ana seride MiniLM sabit; Faz 3'te ablation                  |

**Genel kural:** ölçek serisinde ölçekten başka hiçbir şey değişmeyecek.

## 0.4 — Metrik gereksizliği (raporlamada dikkat)

SQuAD'da sorgu başına ~1 gold doküman olduğu için:
- `Precision@10 ≈ Recall@10 / 10` (neredeyse birebir bağımlı)
- `nDCG@10`, `MRR`'ın monoton dönüşümü

Dört metrik raporlanıyor ama efektif olarak iki bağımsız bilgi var. Tezde bu
açıkça belirtilmeli. Çok-ilgili (multi-relevant) korpusa geçilirse — BEIR
veri setlerinin çoğu böyle — nDCG gerçekten ayrı bilgi taşımaya başlar.

## 0.5 — Ölçüm hijyeni

- **Warm-up:** her DB için ilk 10 sorgu atılacak (lazy load / cache ısınması)
- **Tekrar:** her retrieval pass 3 kez, ortalama ± std
- **İzolasyon:** aynı anda tek DB container'ı ayakta. Şu an `benchmark_db.py`
  altı DB'yi tek process'te koşuyor ve altı container RAM/CPU için yarışıyor
- **Latency:** ortalama yetmez, **p50 / p95** raporla

---

# BÖLÜM 1 — VERİ SETİ SEÇİMİ

## 1.1 — SQuAD v2 tavanı

Benzersiz context sayısı: validation 1.204 + train ~19.035 = **~20.2K tavan**.
`--num-documents 50000` → `ValueError` (`benchmark_db.py:177`).

## 1.2 — Seçenek A: SQuAD içinde kal (Faz 1)

Ölçek eksenini **indekslenen vektör (chunk) sayısı** üzerinden raporla:

| Etiket | num_documents | chunk_size | ~Vektör |
| ------ | ------------- | ---------- | ------- |
| S1     | 100           | 512        | ~190    |
| S2     | 1.000         | 512        | ~1.900  |
| S3     | 10.000        | 512        | ~19.000 |
| S4     | 20.000        | 512        | ~38.000 |
| S5     | 20.000        | 256        | ~70.000 |

Avantaj: yeni kod yok, hemen koşulur.
Dezavantaj: "50K doküman" değil "50K vektör"; tek korpus.

## 1.3 — Seçenek B: BEIR / MS MARCO'ya geç (Faz 2 — ÖNERİLEN)

BEIR, 18 public retrieval veri setini `{corpus, queries, qrels}` JSONL formatında
standartlaştırır. Mevcut `DatasetLoader` ABC'sine yeni bir loader yazmak birkaç
saatlik iş.

| Korpus               | Doküman       | Test sorgu | Rol                        |
| -------------------- | ------------- | ---------- | -------------------------- |
| SciFact              | 5.183         | 300        | küçük                      |
| ArguAna              | 8.670         | 1.406      | küçük                      |
| FiQA                 | 57.638        | 648        | orta                       |
| TREC-COVID           | 171.332       | 50         | orta-büyük                 |
| CQADupStack          | 457.000       | 13.145     | büyük                      |
| FEVER                | ~5.42M        | 6.666      | çok büyük                  |
| HotpotQA             | ~5.23M        | 7.405      | çok büyük                  |
| DBPedia              | 4.635.922     | 400        | çok büyük                  |
| **MS MARCO passage** | **8.841.823** | **6.980**  | **ölçek serisinin tamamı** |

**Öneri:** MS MARCO'yu ölçek korpusu yap (1K → 10K → 100K → 1M alt örneklem),
SQuAD'ı ikinci korpus olarak koru. İki korpus = "bulgu korpustan bağımsız mı"
sorusuna cevap → tez ciddi güçlenir.

Ek fayda: BEIR qrels'te sorgu başına birden fazla ilgili doküman var → nDCG
gerçekten bilgi taşır (bkz. 0.4).

## 1.4 — Seçenek C: Türkçe korpus (Faz 3 — özgünlük katkısı)

TR-MTEB grubu BEIR'in bir bölümünü Türkçeye çevirmiş: **BiText, MsMarco-TR,
Scifact-TR, Fiqa-TR, NFCorpus-TR, Quora-TR**.

MSKÜ tezinde "Türkçe korpusta DB sıralaması değişiyor mu?" alt bölümü hem özgün
hem jüri için ilgi çekici. Embedding tarafı zaten hazır (`multilingual-e5-large`).

## 1.5 — Tablo / yapısal veri (Faz 2 — bkz. Deney F)

Vector DB, vektörün paragraftan mı tablo satırından mı geldiğini bilmez.
Tablo verisi **tek başına** DB'leri ayrıştırmaz. Değeri, metadata filtreleme ve
hibrit aramayı mümkün kılmasında (Deney F).

| Korpus              | Tablo   | Test sorgu |
| ------------------- | ------- | ---------- |
| NQ-Tables           | 169.898 | 919        |
| OTT-QA              | 419.183 | 2.214      |
| WikiSQL             | 24.241  | —          |
| OpenWikiTable       | —       | —          |
| AIT-QA, MultiHierTT | küçük   | —          |

Not: "Excel dosyası kullanmak" tez açısından anlamsız — dosya formatı sadece I/O.
Önemli olan içeriğin **yapısal** olması (kolon, satır, tip) ve bunun filtre alanı
üretmesi.

## 1.6 — Multimodal: ÖNERİLMİYOR

Vector DB açısından "multimodal" diye bir kategori yok. CLIP'ten çıkan 512
boyutlu vektörle MiniLM'den çıkan 384 boyutlu vektör arasında DB'nin umursadığı
tek fark boyuttur. Multimodal eklemek tezin sorusuna yeni cevap üretmez, sadece
pipeline'a bir encoder ekler.

Arkasındaki **gerçek** değişken zaten test edilebilir: **vektör boyutu ve
embedding uzayının intrinsic dimensionality'si** (bkz. Deney G).

---

# BÖLÜM 2 — DENEYLER

## FAZ 1 — Tezin omurgası (zorunlu)

### DENEY A — Ana ölçek serisi

**Amaç:** Ölçek büyüdükçe DB'ler arası kalite ve hız farkının ortaya çıkışı.
**Tez bölümü:** Bulgular → Ölçek analizi (ana tablo + grafik)

| Parametre | Değer                                            |
| --------- | ------------------------------------------------ |
| DB        | 6 (Chroma, Qdrant, FAISS, Milvus, ES, Weaviate)  |
| Ölçek     | S1–S5                                            |
| Index     | HNSW standart (M=16, efC=200, ef=64)             |
| top_k     | 10 (sabit)                                       |
| Sorgu     | 500 (**sabit set**, S1 dokümanlarından seçilmiş) |
| Tekrar    | 3                                                |

**Koşum:** 5 ölçek × 3 tekrar = 15
**Toplanacak:** MRR, nDCG@{1,3,5,10}, P@K, R@K, p50/p95 ms, indexing süresi, peak RAM, disk

---

### DENEY B — Exact vs Approximate (yaklaşıklık maliyeti)

**Amaç:** HNSW'nin exact aramaya göre kaybettiği recall.
**Tez bölümü:** Bulgular → Yaklaşık indeksleme maliyeti

1. FAISS `flat` ile her ölçekte **exact ground truth** üret
2. Her DB'nin HNSW sonucunu iki referansa karşı ölç:
   - **Recall@10 vs exact** (yaklaşıklık kaybı)
   - **Recall@10 vs gold** (görev başarısı)
3. Farkı ölçeğe karşı çiz

**Ölçek:** S2–S5 (S1'de fark çıkmaz) | **Sorgu:** 500

**Beklenti:** S1–S2'de kayıp ~0, S3'ten itibaren açılır. Eski 10K koşumunda
Chroma 0.670 vs diğerleri 0.675 zaten bu sinyali veriyordu.

---

### DENEY C — ANN parametre duyarlılığı (recall–latency Pareto)

**Amaç:** "X daha hızlı" iddiası aslında bir çalışma noktası seçimidir. Aynı
recall seviyesinde hangi DB daha hızlı — asıl soru bu.
**Tez bölümü:** Bulgular → Parametre duyarlılığı / Tartışma

| DB         | Parametre        | Değerler             |
| ---------- | ---------------- | -------------------- |
| Qdrant     | `hnsw_ef`        | 16, 32, 64, 128, 256 |
| Weaviate   | `ef`             | 16, 32, 64, 128, 256 |
| Chroma     | `hnsw:search_ef` | 16, 32, 64, 128, 256 |
| ES         | `num_candidates` | 10, 25, 50, 100, 250 |
| Milvus     | `ef`             | 16, 32, 64, 128, 256 |
| FAISS HNSW | `efSearch`       | 16, 32, 64, 128, 256 |

**Ölçek:** S4 (veya S5) | **Sorgu:** 500 | **Tekrar:** 3

**Çıktı:** x=latency, y=Recall@10, her DB bir eğri → Pareto sınırı.
Tezin en güçlü tek grafiği.

**Not:** 6×5×3 = 90 retrieval pass ama index bir kez kurulup yeniden kullanılır
(search parametresi index'i değiştirmez) → hızlı biter.

---

### DENEY D — İndeksleme maliyeti ve kaynak kullanımı

**Amaç:** Retrieval dışı operasyonel maliyet. **Ek koşum gerekmez** — Deney A'dan çıkar.
**Tez bölümü:** Bulgular → İndeksleme maliyeti

- `*_add_seconds` (saf indeksleme)
- `*_indexing_total_seconds` (embedding dahil)
- peak RAM (`run_with_resource_stats` zaten topluyor)
- **disk index boyutu** (yeni eklenecek — container volume boyutu)

Ölçek eksenine karşı: lineer mi, superlineer mi?

---

### DENEY E — Chunk boyutu etkisi

S4 (512) ve S5 (256) aynı 20K dokümanı farklı chunk boyutuyla indeksliyor →
**ek koşum gerekmez**, ayrı alt bölüm olarak yazılır.

---

## FAZ 2 — Yüksek getirili genişlemeler

### DENEY F — Filtreli ve hibrit arama (EN AYIRT EDİCİ DENEY)

**Amaç:** Şu ana kadarki bulgu "hepsi aynı". Filtreli aramada **kesinlikle aynı
olmayacaklar** — FAISS'in cevabı bile yok.
**Tez bölümü:** Bulgular → Fonksiyonel yetenek farkları

| DB            | Filtreleme                    | Hibrit (BM25+vektör) |
| ------------- | ----------------------------- | -------------------- |
| FAISS         | Yok (sadece IDSelector)       | Yok                  |
| Chroma        | `where`                       | Yok                  |
| Qdrant        | Payload index + filtered HNSW | Var                  |
| Weaviate      | `where`                       | Var (native)         |
| ElasticSearch | Native, en güçlü              | Var (RRF)            |
| Milvus        | Scalar filter + partition     | Sınırlı              |

**Veri:** NQ-Tables veya OTT-QA (tablolar markdown'a serialize edilir; kolon
adları, satır sayısı, domain → filtre alanı). Alternatif: BEIR korpusuna sentetik
metadata (yıl, kategori, kaynak) eklenip filtre selectivity süpürülür.

**Ölçülecek:**
- Filtre selectivity ∈ {%1, %10, %50, %100} × recall × latency
- Yüksek selectivity'de HNSW graph'ının bozulması (pre-filter vs post-filter)
- Hibrit aramada BM25 katkısı (ES ve Weaviate'in avantajı)

**Not:** Bu deney tek başına ayrı bir bölüm/makale büyüklüğünde. Kapsam kararı
danışmanla netleştirilmeli.

---

### DENEY H — Gerçek büyük ölçek (MS MARCO)

**Amaç:** SQuAD tavanını aşıp gerçek üretim ölçeğine çıkmak.
**Ölçek:** 1K → 10K → 100K → 1M pasaj | **Sorgu:** MS MARCO dev (6.980'den 500 örneklem)

Deney A ile aynı protokol, farklı korpus. İki korpusun sonucu örtüşürse
"bulgu korpustan bağımsız" iddiası kurulabilir — tezin genellenebilirlik katkısı.

**Uyarı:** 1M pasaj için disk ve RAM planlaması gerekir (ES ve Milvus en aç olanlar).

---

## FAZ 3 — Opsiyonel katkılar

### DENEY G — Embedding boyutu ablation (multimodal yerine)

**Amaç:** HNSW'nin recall–latency dengesi vektör boyutuna duyarlı mı?
Sınırlılıklar bölümündeki "tek embedding modeli" maddesini kapatır.

| Boyut | Model                              |
| ----- | ---------------------------------- |
| 384   | `all-MiniLM-L6-v2` (mevcut)        |
| 768   | `all-mpnet-base-v2` veya `e5-base` |
| 1024  | `multilingual-e5-large`            |

Aynı korpus, aynı ölçek (S4), üç boyut. Ucuz ve doğrudan tez sorusuna hizmet eder.

---

### DENEY I — Türkçe korpus

TR-MTEB Türkçe çevirileri (MsMarco-TR, Scifact-TR, Fiqa-TR, NFCorpus-TR, Quora-TR)
+ `multilingual-e5-large`. Soru: **DB sıralaması dile göre değişiyor mu?**

Beklenti: değişmemeli (DB dilden bağımsız), ama bunu göstermek de bir bulgudur ve
Türkçe literatüre katkı sayılır.

---

## ÖNERİLMEYEN

### Multimodal (CLIP + Flickr30K / MS COCO)

Gerekçe için bkz. 1.6. Kapsam kayması riski yüksek, tez sorusuna katkısı düşük.
Boyut etkisi Deney G ile zaten karşılanıyor.

---

# BÖLÜM 3 — SIRA VE ZAMAN

| #   | İş                                         | Faz | Bağımlılık | Süre         |
| --- | ------------------------------------------ | --- | ---------- | ------------ |
| 1   | Kod değişiklikleri (0.1–0.5)               | 1   | —          | 1–2 gün      |
| 2   | S1–S2 smoke test, tutarlılık kontrolü      | 1   | 1          | 1 saat       |
| 3   | **Deney A**                                | 1   | 2          | 1 gün        |
| 4   | **Deney B**                                | 1   | 3          | yarım gün    |
| 5   | **Deney C**                                | 1   | 3          | yarım gün    |
| 6   | Deney D + E analizi                        | 1   | 3          | ek koşum yok |
| 7   | Grafik/tablo üretimi                       | 1   | 3–6        | 1 gün        |
| —   | **Faz 1 burada bitiyor — tez yazılabilir** |     |            |              |
| 8   | BEIR loader yaz                            | 2   | —          | yarım gün    |
| 9   | **Deney H** (MS MARCO)                     | 2   | 8          | 2 gün        |
| 10  | **Deney F** (filtreli/hibrit)              | 2   | 8          | 3–4 gün      |
| 11  | **Deney G** (boyut)                        | 3   | 3          | 1 gün        |
| 12  | **Deney I** (Türkçe)                       | 3   | 8          | 1 gün        |

**Kritik uyarı:** Faz 1 bitmeden Faz 2/3'e geçme. Faz 1 tek başına savunulabilir
bir tez üretiyor; Faz 2 ve 3 katkıyı artırır ama bitirmeme riskini de artırır.

---

# BÖLÜM 4 — GEÇERSİZ KILINACAK ESKİ SONUÇLAR

`archive/`'e taşınacak, tezde kullanılmayacak:

- `official_scale_10000docs_4db_20260525_120926.json`
  → Milvus Lite, Qdrant in-memory, Weaviate/ES yok, 200 sorgu
- `official_scale_1000docs_4db_20260524_171541.json` → aynı
- `official_scale_50docs_4db_20260524_170216.json` → aynı

`official_scale_*_6db_GPU_realserver_*` gerçek sunucuyla üretilmiş ama **index
tipi standardize edilmeden** → Deney A ile yenilenecek.

Baseline serisi (`official_baseline_*_topk10_*`) 6 DB × gerçek sunucu ile doğru,
ancak index standardizasyonu sonrası tutarlılık için tekrar koşulması önerilir.

---

# BÖLÜM 5 — SINIRLILIKLAR BÖLÜMÜNE YAZILACAKLAR

1. **Korpus tavanı:** SQuAD v2'de ~20.2K benzersiz context var; daha büyük
   ölçek chunk boyutu küçültülerek vektör sayısı üzerinden elde edildi
   (Faz 2 yapılırsa bu madde MS MARCO ile kapanır)
2. **Milvus Lite bulgusu:** `in_memory=True` embedded mod ~10x yavaş
3. **ES/Weaviate numpy mock:** eski `in_memory=True` gerçek DB davranışını temsil etmiyordu
4. **Metrik bağımlılığı:** SQuAD'da sorgu başına ~1 gold doküman olduğundan
   P@10 ≈ R@10/10 ve nDCG@10 ≈ MRR'ın monoton dönüşümü
5. **Sabit sorgu seti:** en küçük ölçeğin dokümanlarından seçildi; ölçek etkisini
   izole eder ancak sorgu dağılımını o alt kümeye sabitler
6. **Tek makine, izole test:** dağıtık/çok-node senaryosu kapsam dışı
7. **Tek embedding modeli** (Faz 3 yapılırsa kapanır)
8. **Retrieval-odaklı kapsam:** generation değerlendirmesi kapsam dışı

---

# EK — KAYNAK NOTLARI (kaynakçaya eklenecek)

- Thakur ve ark. (2021) — BEIR benchmark
- Bajaj ve ark. (2016) — MS MARCO
- Herzig ve ark. (2021) — NQ-Tables
- Chen ve ark. (2020/2021) — OTT-QA
- Kwiatkowski ve ark. (2019) — Natural Questions
- Baysan & Güngör (2025) — TR-MTEB (Türkçe retrieval)
- Malkov & Yashunin (2020) — HNSW (mevcut)