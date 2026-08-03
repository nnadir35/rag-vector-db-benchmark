# Tez Deney Planı — RAG Vector DB Benchmark

Danışman kararı: **10K/50K ölçekte kalite farkı gösterilecek.**

Korpus: **MS MARCO passage** (8.841.823 pasaj, 6.980 dev sorgu). Sorgu seti,
gold pasajı taban ölçekte (S1) bulunan sorgulardan seçilen, tekrarlanabilir
seed'li (seed=42) 500 sorguluk sabit bir örneklem.

---

# BÖLÜM 0 — Deneylerden ÖNCE yapılacak kod ön koşulları

Bunlar yapılmadan koşulan hiçbir sonuç tezde kullanılamaz. Kod tabanında
hazır ve doğrulanmış durumda.

## 0.1 — Index tipi standardizasyonu (KRİTİK) — ✅ hazır

| DB            | Index                                                                          | Arama tipi          |
| ------------- | ------------------------------------------------------------------------------ | ------------------- |
| Chroma        | HNSW, M=16 / construction_ef=200 / search_ef=64                                | Approximate         |
| Qdrant        | HNSW, m=16 / ef_construct=200 / hnsw_ef=64                                     | Approximate         |
| Milvus        | HNSW (aynı parametreler); ayrıca FLAT modu ground-truth üretimi için korunuyor | Approximate / Exact |
| ElasticSearch | HNSW, index_options m=16/ef_construction=200, num_candidates config'den        | Approximate         |
| Weaviate      | HNSW, max_connections=16/ef_construction=200/ef=64                             | Approximate         |

**FAISS sistematik benchmark'tan hariç.** Gerekçe: FAISS in-process bir
kütüphane, ağ katmanı yok; Docker üzerinde çalışan diğer beş DB ile latency
karşılaştırması "hepsi aynı şartlarda" ilkesini bozar. Kod tabanında adaptör
olarak duruyor (Pinecone ile aynı desen — mevcut ama sistematik koşuma dahil
değil) ve Deney B'de ground truth referansı **Milvus FLAT** ile üretiliyor
(zaten Docker'da, ilkeyi bozmuyor).

## 0.2 — Sabit sorgu seti (KRİTİK) — ✅ hazır, MS MARCO'ya özel doğrulama gerekli

Pasaj koleksiyonu `pid` sırasına göre prefix alınarak ölçekleniyor
(S1 ⊂ S2 ⊂ S3 ⊂ S4 ⊂ S5 ⊂ S6), böylece küçük ölçeğin korpusu büyüğünün alt
kümesi oluyor. Sorgu seti en küçük ölçekten (S1, 1.000 pasaj) seçilip
sabitleniyor, tüm ölçeklerde aynısı kullanılıyor — ölçekler arası recall
düşüşü saf distractor etkisi oluyor, sorgu setinin değişmesinden kaynaklı
bir karışıklık (confound) olmuyor.

**Loader yazılırken doğrulanacak:** S1 içinde en az 500 sorgunun gold
pasajı bulunmalı. Bulunmuyorsa S1'i büyütmek ya da seçim stratejisini
revize etmek gerekir — bu, kod yazımının ilk kontrol adımı.

## 0.3 — Değişken sınıflandırması — ✅ hazır

| Değişken         | Türü                                    | Karar                                                                    |
| ---------------- | --------------------------------------- | ------------------------------------------------------------------------ |
| `top_k`          | Treatment — bulguyu doğrudan değiştirir | 10 sabit, ayrı ablation                                                  |
| `num_queries`    | Precision — hata payını değiştirir      | 500 sabit                                                                |
| Tekrar sayısı    | Noise — varyans tahmini                 | 3 sabit                                                                  |
| Embedding modeli | Treatment                               | Ana seride sabit; ayrı ablation olarak Deney G'de sistematik test edilir |

**Not:** MS MARCO pasajları zaten kısa segmentler (ortalama ~55 kelime) —
ek chunklamaya ihtiyaç yok, bir pasaj = bir vektör. Bu yüzden chunk_size
bu planda bir değişken olarak yer almıyor.

## 0.4 — Metrik gereksizliği — doğrulanmalı, varsayılmamalı

MS MARCO dev seti tipik olarak sorgu başına ~1 ilgili pasaj etiketliyor
(sparse qrels). Bu doğruysa `Precision@10 ≈ Recall@10 / 10` ve `nDCG@10`,
`MRR`'ın monoton bir dönüşümü olur — dört metrik raporlanır ama efektif iki
bağımsız bilgi taşınır. Bu, loader yazıldıktan sonra Deney A'nın ilk
koşumunda ölçülerek doğrulanmalı.

## 0.5 — Ölçüm hijyeni — ✅ hazır

- Warm-up: her DB için ilk 10 sorgu atılıyor
- Her retrieval pass 3 kez koşuluyor, ortalama ± std raporlanıyor
- `clear()` — her DB'nin koleksiyon temizleme mantığı koşulsuz sunucu-taraflı
  silme yapıyor (instance-state'e bağlı değil); her ölçek koşumundan
  önce/sonra kayıt sayısı loglanıp beklenen chunk sayısıyla karşılaştırılıyor
- Latency **ayrıştırılmış** ölçülüyor: `search_only_*_ms` (sadece arama,
  `retrieve_with_embedding` ile) ve `query_embedding_avg_ms` (sorgu
  embedding maliyeti) ayrı raporlanıyor — birleşik ölçüm sabit embedding
  maliyetinin gerçek arama-süresi büyümesini maskelemesini önlüyor
- p50/p95 latency raporlanıyor, sadece ortalama değil
- **Açık nokta:** DB izolasyonu — altı container tek process'ten sırayla
  koşuluyor, aynı anda RAM/CPU için yarışıyorlar. MS MARCO'nun büyük
  ölçeklerinde (100K-200K pasaj) bu SQuAD ölçeğindekinden daha ciddi bir
  gürültü kaynağı olabilir; Deney A başlamadan danışmanla değerlendirilmeli

---

# BÖLÜM 1 — VERİ SETİ

**MS MARCO passage** (Bajaj ve ark., 2016), BEIR standardında
`{corpus, queries, qrels}` JSONL formatında. `DatasetLoader` ABC'sine yeni
bir `MSMARCOLoader` yazılacak (mevcut retriever/config deseniyle tutarlı).

| Korpus           | Doküman   | Test sorgu  |
| ---------------- | --------- | ----------- |
| MS MARCO passage | 8.841.823 | 6.980 (dev) |

**Ölçek serisi:**

| Etiket | Pasaj sayısı | Not                                                  |
| ------ | ------------ | ---------------------------------------------------- |
| S1     | 1.000        | Sabit sorgu setinin seçildiği taban ölçek            |
| S2     | 10.000       |                                                      |
| S3     | 20.000       |                                                      |
| S4     | 50.000       | Danışmanın "50K" hedefini doğrudan karşılıyor        |
| S5     | 100.000      | Embedder ablation'ının (Deney G) da koşulacağı ölçek |
| S6     | 200.000      | Üst ölçek                                            |

**Geçiş kontrol listesi (loader yazılırken):**
- [ ] `MSMARCOLoader`, `DatasetLoader` ABC'sine uyuyor
- [ ] Pasaj prefix mantığı (S1⊂S2⊂S3⊂S4⊂S5⊂S6) kuruluyor
- [ ] S1'de en az 500 sorgunun gold pasajı var (bkz. 0.2)
- [ ] `_select_queries_for_documents()` MS MARCO qrels formatına genelleştirildi
- [ ] `benchmark_all_dbs.yaml`'a `dataset: msmarco` seçeneği eklendi

**Tek korpus notu:** Sonuçlar yalnızca MS MARCO passage korpusu için
geçerli olacak. Genellenebilirlik iddiası bu planda kurulmuyor; istenirse
ileride ikinci bir korpus (BEIR'in başka bir alt kümesi) eklenerek
"bulgu korpustan bağımsız mı" sorusu ayrı bir uzantı olarak ele alınabilir.

**Türkçe uzantısı (Faz 3, Deney I):** TR-MTEB'in MsMarco-TR çevirisi, aynı
kaynak korpusun Türkçe versiyonu olduğu için doğrudan karşılaştırılabilir
bir ikinci veri noktası sunuyor — bkz. Deney I.

---

# BÖLÜM 2 — DENEYLER

## FAZ 1 — Tezin omurgası (zorunlu)

### DENEY A — Ana ölçek serisi

**Amaç:** Ölçek büyüdükçe DB'ler arası kalite ve hız farkının ortaya çıkışı.
**Tez bölümü:** Bulgular → Ölçek analizi (ana tablo + grafik)

| Parametre | Değer                                               |
| --------- | --------------------------------------------------- |
| DB        | 5 (Chroma, Qdrant, Milvus, ElasticSearch, Weaviate) |
| Ölçek     | S1–S6 (1K / 10K / 20K / 50K / 100K / 200K pasaj)    |
| Index     | HNSW standart (M=16, efC=200, ef=64)                |
| top_k     | 10 (sabit)                                          |
| Sorgu     | 500 (sabit set, S1'den seçilmiş)                    |
| Tekrar    | 3                                                   |

**Koşum:** 6 ölçek × 3 tekrar = 18
**Toplanacak:** MRR, nDCG@{1,3,5,10}, P@K, R@K, search_only p50/p95 ms,
query_embedding_ms, indexing süresi, peak RAM, disk boyutu

---

### DENEY B — Exact vs Approximate (yaklaşıklık maliyeti)

**Amaç:** HNSW'nin exact aramaya göre kaybettiği recall — tezin en özgün katkısı.

1. **Milvus FLAT** ile her ölçekte exact ground truth üret
2. Her DB'nin HNSW sonucunu iki referansa karşı ölç:
   - Recall@10 vs exact (yaklaşıklık kaybı)
   - Recall@10 vs gold (görev başarısı)
3. Farkı ölçeğe karşı çiz

**Ölçek:** S2–S6 (S1'de fark çıkmaz) | **Sorgu:** 500

**Beklenti:** S1–S2'de kayıp ~0, S3'ten itibaren açılır.

---

### DENEY C — ANN parametre duyarlılığı (recall–latency Pareto)

**Amaç:** "X daha hızlı" iddiası aslında bir çalışma noktası seçimidir. Aynı
recall seviyesinde hangi DB daha hızlı — asıl soru bu.

| DB       | Parametre        | Değerler             |
| -------- | ---------------- | -------------------- |
| Qdrant   | `hnsw_ef`        | 16, 32, 64, 128, 256 |
| Weaviate | `ef`             | 16, 32, 64, 128, 256 |
| Chroma   | `hnsw:search_ef` | 16, 32, 64, 128, 256 |
| ES       | `num_candidates` | 10, 25, 50, 100, 250 |
| Milvus   | `ef`             | 16, 32, 64, 128, 256 |

**Ölçek:** S5 (veya S6) | **Sorgu:** 500 | **Tekrar:** 3

**Çıktı:** x=search_only latency, y=Recall@10, her DB bir eğri → Pareto
sınırı. Tezin en güçlü tek grafiği.

**Not:** 5×5×3 = 75 retrieval pass ama index bir kez kurulup yeniden
kullanılır (arama parametresi index'i değiştirmez) → hızlı biter.

---

### DENEY D — İndeksleme maliyeti ve kaynak kullanımı

**Amaç:** Retrieval dışı operasyonel maliyet. Ek koşum gerekmez — Deney A'dan çıkar.

- `*_add_seconds` (saf indeksleme)
- `*_indexing_total_seconds` (embedding dahil)
- peak RAM
- disk index boyutu

Ölçek eksenine karşı: lineer mi, superlineer mi? S5/S6 (100K-200K pasaj)
ölçeğinde bu eğrinin şekli gerçek anlamda büyük-ölçek davranışını
gösterebilecek büyüklükte.

---

## FAZ 2 — Yüksek getirili genişlemeler

### DENEY F — Filtreli ve hibrit arama (EN AYIRT EDİCİ DENEY)

**Amaç:** Deney A'nın bulgusu "hepsi birbirine yakın" ise, filtreli aramada
kesinlikle aynı olmayacaklar.

| DB            | Filtreleme                    | Hibrit (BM25+vektör) |
| ------------- | ----------------------------- | -------------------- |
| Chroma        | `where`                       | Yok                  |
| Qdrant        | Payload index + filtered HNSW | Var                  |
| Weaviate      | `where`                       | Var (native)         |
| ElasticSearch | Native, en güçlü              | Var (RRF)            |
| Milvus        | Scalar filter + partition     | Sınırlı              |

**Veri:** MS MARCO'ya sentetik metadata (yıl, kategori, kaynak) eklenip
filtre selectivity süpürülür — MS MARCO'nun kendisinde doğal filtre alanı yok.

**Ölçülecek:**
- Filtre selectivity ∈ {%1, %10, %50, %100} × recall × latency
- Yüksek selectivity'de HNSW graph'ının bozulması (pre-filter vs post-filter)
- Hibrit aramada BM25 katkısı

**Not:** Bu deney tek başına ayrı bir bölüm büyüklüğünde. Kapsam kararı
danışmanla netleştirilmeli.

---

## FAZ 3 — Opsiyonel katkılar

### DENEY G — Embedder Ablation (detaylı)

**"Ablation" ne demek?** Bir sistemin bir bileşenini değiştirip/çıkarıp
geri kalan her şeyi sabit tutarak, o bileşenin sonuca katkısını izole
etme yöntemi. Burada bileşen = embedding modeli. DB, HNSW parametreleri,
ölçek, sorgu seti hep aynı kalıyor, sadece embedder değişiyor — çıkan
fark saf embedder etkisi oluyor, başka hiçbir şeyle karışmıyor.

**Amaç:** Deney A, DB seçiminin retrieval kalitesine etkisinin küçük
olduğunu gösterebilir (tipik HNSW-standardize edilmiş karşılaştırmalarda
beklenen bir sonuç). Bu durumda tezin doğal bir sonraki sorusu: **madem DB
önemli değil, ne önemli?** Cevap büyük ihtimalle embedder. Bu deney bunu
sistematik ve ölçülü şekilde göstermek için var.

**Beklenen etki büyüklüğü:** Genel RAG literatüründe embedder seçiminin
tek başına retrieval doğruluğunu %20-30 oranında değiştirebildiği
bildiriliyor — bu, Deney A'daki DB'ler-arası farktan (~%1-2) bir büyüklük
mertebesi fazla. Açık ağırlıklı en iyi modeller (örn. Qwen3-Embedding,
NV-Embed) günümüzde ticari API'lerin en iyilerini (OpenAI
text-embedding-3-large) MTEB'de 7-10 puan geçebiliyor. Bu rakamlar genel
literatür eğilimi olarak alınmalı, MS MARCO'ya özgü kesin sayı olarak değil
— asıl sayı bu deneyle üretilecek.

**İki ayrı eksen — karıştırılmamalı:**

1. **Boyut ekseni:** aynı eğitim rejiminden, sadece vektör boyutu farklı
   modeller. HNSW'nin arama karmaşıklığı ve bellek/disk kullanımı boyutla
   değişir.
2. **Eğitim rejimi/nesil ekseni:** 2021'in genel-amaçlı SBERT modelleri
   (all-MiniLM, all-mpnet) vs 2023+'ın retrieval'a özel contrastive
   eğitilmiş modelleri (e5, bge, gte) vs güncel nesil (Qwen3-Embedding,
   BGE-M3, NV-Embed). Bu eksen boyuttan bağımsız çalışır ve genelde çok
   daha büyük fark yaratır.

**Test edilecek modeller:**

| Model                              | Boyut | Nesil / aile                       | Lisans     | Rolü                                                                                                   |
| ---------------------------------- | ----- | ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| all-MiniLM-L6-v2                   | 384   | 2021 SBERT, damıtılmış             | Apache 2.0 | Taban çizgi (Deney A'da da kullanılan)                                                                 |
| e5-base-v2 veya bge-base-en-v1.5   | 768   | 2023 contrastive, retrieval'a özel | MIT        | Eğitim rejimi ekseni — taban çizgiye yakın boyutta, farklı nesil                                       |
| multilingual-e5-large              | 1024  | 2023 contrastive, çok dilli        | MIT        | Boyut + dil kapsamı ekseni                                                                             |
| BGE-M3                             | 1024  | 2024, dense+sparse+multi-vector    | MIT        | 2026'da self-hosted prod RAG'de yaygın varsayılan; Deney F'nin hibrit arama bulgularıyla da bağlantılı |
| *(opsiyonel)* Qwen3-Embedding-0.6B | 1024  | 2026 güncel nesil, açık ağırlıklı  | Apache 2.0 | Üst sınır — en yeni neslin ne kadar iyileştirdiği                                                      |

**Neden bu modeller, neden bu sırayla:**
- all-MiniLM → e5-base/bge-base: benzer boyutta, farklı eğitim rejimi —
  fark büyükse "eğitim rejimi boyuttan daha belirleyici" bulgusu kurulur
- e5-base → multilingual-e5-large: aynı aile, boyut artışı + çok-dillilik
  birlikte değişiyor; ikisini ayırmak istenirse araya `e5-large-v2`
  (İngilizce, 1024 boyut) eklenip boyut sabit tutularak sadece dil kapsamı
  izole edilebilir
- multilingual-e5-large → BGE-M3: aynı boyut mertebesi, farklı mimari
  (BGE-M3 tek modelde dense+sparse+multi-vector üretebiliyor)

**Neden API tabanlı modeller (OpenAI, Gemini, Cohere) dahil değil:**
İki sorun yaratıyorlar: (1) ağ gecikmesi embedding ölçümüne karışır — tam
olarak Bölüm 0.5'te ayırdığımız `search_only`/`query_embedding_ms`
temizliğini bozar; (2) internet bağımlılığı ve API maliyeti
tekrarlanabilirliği zorlaştırır. İstenirse Faz 3'e opsiyonel bir "ticari
üst sınır" karşılaştırması olarak eklenebilir, zorunlu değil.

**Metodoloji:**
1. Tek ölçek üzerinde koş: **S5 (100.000 pasaj)** — tüm 6 ölçekte × 5
   model kombinasyonu tez zaman bütçesini aşar
2. 5 DB de dahil, sabit HNSW parametreleri, sabit 500 sorgu seti korunur
3. Her embedder için: (a) S5 korpusunu bu modelle yeniden embed et, (b)
   her DB'ye aynı HNSW parametreleriyle yeniden indeksle, (c) aynı 500
   sorguyu bu modelle embed et, (d) recall + search_only latency ölç
4. `query_embedding_avg_ms` her model için ayrı raporlanır — modelin
   kendi hızı da bir bulgu (büyük modeller daha yavaş embed eder, bu
   üretim ortamında bir trade-off)
5. Boyut ekseni ile eğitim-rejimi ekseni ayrı ayrı yorumlanır, tek bir
   "en iyi model" sonucuna indirgenmez

**Çıktı tablosu (Deney A ile aynı formatta, embedder ekseni eklenmiş):**

| Embedder | Boyut | DB  | Recall@10 | search_only p50 | query_embedding_ms |
| -------- | ----- | --- | --------- | --------------- | ------------------ |

**Süre/kaynak notu:** Her embedder = S5'in tam yeniden embed edilmesi +
5 DB'ye yeniden indeksleme. Bu, Deney A'nın S5 koşumunun ~5 katı bir iş
yükü demek (4-5 model). İlk model koşulduktan sonra gerçek süre ölçülüp
kalan modeller için tahmin güncellenmeli — Bölüm 0.5'teki tahmin
formülüyle aynı disiplin.

**Kapsam notu:** Bu deneyin etkisi büyük çıkması muhtemel olduğu için
Faz 1'e (zorunlu omurga) çekilip çekilmeyeceği danışmanla görüşülebilir.
Şu an Faz 3'te (opsiyonel) duruyor çünkü Bölüm 3'ün genel kuralı geçerli:
Faz 1 bitmeden Faz 2/3'e geçilmemeli.

---

### DENEY I — Türkçe korpus

TR-MTEB'in MsMarco-TR çevirisi + `multilingual-e5-large` (Deney G'de zaten
test edilen model). Soru: **DB sıralaması dile göre değişiyor mu?**

Beklenti: değişmemeli (DB dilden bağımsız), ama bunu göstermek de bir
bulgudur ve Türkçe literatüre katkı sayılır.

---

## ÖNERİLMEYEN

### Multimodal

Vector DB açısından "multimodal" diye bir kategori yok — CLIP'ten çıkan
512 boyutlu vektörle MiniLM'den çıkan 384 boyutlu vektör arasında DB'nin
umursadığı tek fark boyuttur. Bu zaten Deney G'nin boyut ekseninde
karşılanıyor; ayrı bir encoder eklemek tezin sorusuna yeni bir cevap
üretmüyor.

---

# BÖLÜM 3 — SIRA VE ZAMAN

| #   | İş                                                    | Faz | Bağımlılık | Süre                                                      |
| --- | ----------------------------------------------------- | --- | ---------- | --------------------------------------------------------- |
| 1   | Kod ön koşulları (0.1–0.5)                            | 1   | —          | ✅ hazır                                                   |
| 2   | MS MARCO / BEIR loader yaz                            | 1   | 1          | yarım gün                                                 |
| 3   | Geçiş kontrol listesi (Bölüm 1) doğrulaması           | 1   | 2          | 1-2 saat                                                  |
| 4   | S1 smoke test + sabit sorgu seti riski kontrolü (0.2) | 1   | 2-3        | 1 saat                                                    |
| 5   | **Deney A** (S1-S6)                                   | 1   | 4          | 1-2 gün — ilk gerçek süre S1-S2'den sonra tahmin edilecek |
| 6   | **Deney B**                                           | 1   | 5          | yarım gün                                                 |
| 7   | **Deney C**                                           | 1   | 5          | yarım gün                                                 |
| 8   | Deney D analizi                                       | 1   | 5          | ek koşum yok                                              |
| 9   | Grafik/tablo üretimi                                  | 1   | 5-8        | 1 gün                                                     |
| —   | **Faz 1 burada bitiyor — tez yazılabilir**            |     |            |                                                           |
| 10  | **Deney F** (filtreli/hibrit)                         | 2   | —          | 3-4 gün                                                   |
| 11  | **Deney G** (embedder ablation)                       | 3   | 5          | 2-3 gün (4-5 model × yeniden embed + yeniden indeksleme)  |
| 12  | **Deney I** (Türkçe)                                  | 3   | 2, 11      | 1 gün                                                     |

**Kritik uyarı:** Faz 1 bitmeden Faz 2/3'e geçme. Faz 1 tek başına
savunulabilir bir tez üretiyor; Faz 2 ve 3 katkıyı artırır ama bitirmeme
riskini de artırır.

---

# BÖLÜM 4 — SINIRLILIKLAR BÖLÜMÜNE YAZILACAKLAR

1. **Tek korpus:** Sonuçlar yalnızca MS MARCO passage korpusu için geçerli;
   "bulgu korpustan bağımsız mı" sorusu bu planda kurulmuyor
2. **Milvus Lite / mock modlar:** `in_memory=True` embedded modların
   (Milvus Lite, ES/Weaviate numpy mock) gerçek sunucu davranışını
   temsil etmediği doğrulandı — sistematik koşumda kullanılmıyor
3. **Metrik bağımlılığı:** MS MARCO dev seti sparse qrels kullanıyorsa
   P@10≈R@10/10 ve nDCG@10≈MRR'ın monoton dönüşümü olabilir — doğrulanacak
4. **Sabit sorgu seti:** en küçük ölçeğin (S1, 1K pasaj) gold'una sahip
   sorgulardan seçildi; ölçek etkisini izole eder ama sorgu dağılımını
   o alt kümeye sabitler
5. **Tek makine, izole test:** dağıtık/çok-node senaryosu kapsam dışı;
   MS MARCO'nun büyük ölçeklerinde (100K-200K) DB izolasyonu eksikliği
   SQuAD ölçeğindekinden daha görünür bir gürültü kaynağı olabilir
6. **Embedder ablation'ı tek ölçekte (S5):** tüm ölçeklerde koşulmadı,
   zaman bütçesi nedeniyle — boyut/nesil etkisinin ölçekle nasıl
   değiştiği bu planda ayrı bir soru olarak kalıyor
7. **Retrieval-odaklı kapsam:** generation değerlendirmesi kapsam dışı

---

# EK — KAYNAK NOTLARI (kaynakçaya eklenecek)

- Thakur ve ark. (2021) — BEIR benchmark
- Bajaj ve ark. (2016) — MS MARCO
- Malkov & Yashunin (2020) — HNSW
- Baysan & Güngör (2025) — TR-MTEB (Türkçe retrieval)
- MTEB Leaderboard (Muennighoff ve ark., 2023) — embedder karşılaştırması
  için güncel liderlik tablosu, Deney G yazılırken tekrar kontrol edilmeli
  (liderlik sık değişiyor)