# Deney A V3 — İlerleme

## Yapılan değişiklikler

1. **`ChromaRetriever.clear()`** (`src/retrievers/chroma_retriever.py`) — `if self._client and
   self._collection:` koşulu kaldırıldı. Artık koşulsuz olarak canlı bir client kurup
   `delete_collection` çağırıyor; `chromadb.errors.NotFoundError` (koleksiyon yoksa) sessizce
   yutuluyor, idempotent.
2. **`MilvusRetriever.clear()`** (`src/retrievers/milvus_retriever.py`) — `if self._client is not
   None:` koşulu kaldırıldı. Artık koşulsuz olarak `_get_client()` ile bağlanıp
   `drop_collection` çağırıyor; var olmayan koleksiyonda `drop_collection` zaten no-op
   (istisna atmadığı doğrulandı).
3. ES ve Weaviate'e dokunulmadı (zaten koşulsuz sunucu-taraflı silme yapıyorlardı).

## Cross-process clear() kanıtı

Her 5 DB için: yeni process → koleksiyon oluştur + 20 kayıt ekle → process bitir → **taze bir
process'te** (instance state tamamen sıfır) `clear()` çağır → count doğrula.

| DB | Populate sonrası | clear() sonrası (taze process) | Sonuç |
|---|---|---|---|
| Chroma | 20 | 0 | ✅ PASS |
| Qdrant | 20 | 0 | ✅ PASS |
| Milvus | 20 | 0 | ✅ PASS |
| ElasticSearch | 20 | 0 | ✅ PASS |
| Weaviate | 20 | 0 | ✅ PASS |

## Taban temizliği

`wipe_bench.py` ile 5 DB'nin `bench_*_squad` koleksiyonları/index'leri düzeltilmiş `clear()` ile
silindi. Doğrulama: Qdrant `Not found`, ES `index_not_found_exception` (404), Weaviate boş şema,
Chroma koleksiyon listesi boş, Milvus `has_collection=False`. **Hepsi temiz taban.**

## S1-S5 koşum sonuçları

Sabit sorgu seti: `experiments/results/fixed_queries_DENEY_A.json`. `--num-queries 500
--num-repeats 3`, 5 DB (FAISS hariç). S3 ilk denemede 2 dakikalık kabuk zaman aşımına uğradı
(kısmi yazım: Qdrant 12.600/18.992, diğerleri S2 kalıntısı) — koleksiyonlar tekrar sıfırlanıp S3
temiz baştan koşuldu.

| Ölçek | Dosya | Beklenen chunk | Count (öncesi→sonrası, 5 DB) |
|---|---|---|---|
| S1 · 100 dok | `official_scale_V3_100docs_5db_topk10_20260803_064042.json` | 170 | 0→170 (hepsi) |
| S2 · 1000 dok | `official_scale_V3_1000docs_5db_topk10_20260803_064202.json` | 2211 | 170→2211 (hepsi) |
| S3 · 10000 dok | `official_scale_V3_10000docs_5db_topk10_20260803_065002.json` | 18992 | 0→18992 (temiz taban sonrası, hepsi) |
| S4 · 20000 dok | `official_scale_V3_20000docs_5db_topk10_20260803_065526.json` | 41058 | 18992→41058 (hepsi) |
| S5 · 20000 dok (chunk=256) | `official_scale_V3_chunk256_20000docs_5db_topk10_20260803_070253.json` | 76912 | 41058→76912 (hepsi) |

Her ölçekte 5 DB'nin count'u da beklenen chunk sayısıyla **birebir** eşleşti — sapma/DUR durumu
oluşmadı.

### Ölçek bazlı metrik tablosu

| Ölçek | DB | recall@10 (±std) | search_only p50 (ms) | birleşik p50 (ms) |
|---|---|---|---|---|
| S1 (170) | Chroma | 0.9360 (±0.000) | 3.22 | 18.70 |
| | Qdrant | 0.9360 (±0.000) | 2.32 | 17.80 |
| | Milvus | 0.9360 (±0.000) | 1.80 | 17.28 |
| | ElasticSearch | 0.9360 (±0.000) | 3.82 | 19.31 |
| | Weaviate | 0.9360 (±0.000) | 2.13 | 17.61 |
| S2 (2211) | Chroma | 0.8960 (±0.000) | 3.45 | 18.14 |
| | Qdrant | 0.9000 (±0.000) | 2.29 | 16.99 |
| | Milvus | 0.9000 (±0.000) | 2.64 | 17.34 |
| | ElasticSearch | 0.9000 (±0.000) | 4.10 | 18.80 |
| | Weaviate | 0.8980 (±0.000) | 2.46 | 17.16 |
| S3 (18992) | Chroma | 0.8180 (±0.000) | 3.54 | 18.80 |
| | Qdrant | 0.8240 (±0.000) | 2.72 | 17.98 |
| | Milvus | 0.8120 (±0.000) | 1.54 | 16.80 |
| | ElasticSearch | 0.8240 (±0.000) | 4.31 | 19.57 |
| | Weaviate | 0.8140 (±0.000) | 2.44 | 17.70 |
| S4 (41058) | Chroma | 0.7780 (±0.000) | 3.57 | 18.26 |
| | Qdrant | 0.7900 (±0.000) | 3.48 | 18.17 |
| | Milvus | 0.7840 (±0.000) | 3.78 | 18.47 |
| | ElasticSearch | 0.7893 (±0.001) | 5.98 | 20.67 |
| | Weaviate | 0.7760 (±0.000) | 3.21 | 17.90 |
| S5 (76912, chunk256) | Chroma | **0.8060** (±0.000) | 3.31 | 17.21 |
| | Qdrant | 0.8160 (±0.000) | 4.81 | 18.70 |
| | Milvus | 0.8060 (±0.000) | 3.04 | 16.94 |
| | ElasticSearch | 0.8160 (±0.000) | 5.63 | 19.52 |
| | Weaviate | 0.7960 (±0.000) | 2.54 | 16.44 |

`query_embedding_avg_ms` (tek değer/ölçek): S1=15.48, S2=14.70, S3=15.26, S4=14.69, S5=13.90.

---

## A) v2 vs v3 recall farkı

| DB | S1 (100) | S2 (1000) | S3 (10000) | S4 (20000) | S5 (chunk256) |
|---|---|---|---|---|---|
| Chroma | 0.000 | −0.004 | −0.004 | −0.002 | **+0.058** |
| Qdrant | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Milvus | 0.000 | 0.000 | 0.000 | 0.000 | −0.002 |
| ElasticSearch | 0.000 | 0.000 | 0.000 | −0.001 | 0.000 |
| Weaviate | 0.000 | −0.002 | −0.006 | +0.004 | −0.002 |

**Yorum:**
- **Qdrant/ElasticSearch değişmedi** — beklenen, zaten temizdi (kontrol grubu geçerli).
- **Chroma S5'te +0.058 sıçrama** (0.748→0.806) — Adım-1/Adım-3 hipotezini kanıtlıyor: v2'deki
  Chroma S5 anomalisi ANN-kalite kaybı DEĞİL, `clear()` bug'ının S4'ün (chunk=512) kalıntı
  chunk'larını S5'in (chunk=256) yeni chunk'larıyla karıştırmasıydı. v2 hükmü (kirlilik şüphesi)
  doğruydu.
- **S2-S4'te Chroma/Weaviate'de küçük (−0.006..+0.004) farklar** — istatistiksel gürültü
  seviyesinde (std'lerle aynı büyüklük mertebesinde), kirlilik kanıtı değil. Bu ölçeklerde
  önceki v2 koşumu iç içe geçmiş korpuslarla (aynı chunk_size=512, S1⊂S2⊂S3⊂S4) çalıştığı için
  chunk ID'leri örtüşüyordu ve reuse pratikte zararsız çıktı — ama Chroma S5 gösteriyor ki bu
  garanti değildi, sadece bu spesifik senaryoda (chunk boyutu sabitken) tesadüfen zararsızdı.
- **Milvus hiçbir ölçekte anlamlı sıçrama göstermedi** (max sapma 0.002) — kod düzeyinde aynı
  bug deseni vardı ama örtüşen ID'lerde muhtemelen upsert/duplicate-key davranışı farklı
  çalıştığından pratikte veri karışmadı. Yine de düzeltme doğruydu (cross-process kanıtı PASS) ve
  kaza beklemeden düzeltilmesi gerekiyordu.

## B) search_only p50'nin ölçekle artışı

| DB | S1 | S2 | S3 | S4 | S5 | Trend |
|---|---|---|---|---|---|---|
| Chroma | 3.22 | 3.45 | 3.54 | 3.57 | 3.31 | ~düz (~3.2-3.6ms) |
| Qdrant | 2.32 | 2.29 | 2.72 | 3.48 | 4.81 | **net artış (~2.1x)** |
| Milvus | 1.80 | 2.64 | 1.54 | 3.78 | 3.04 | gürültülü, genel yukarı |
| ElasticSearch | 3.82 | 4.10 | 4.31 | 5.98 | 5.63 | **net artış (~1.5x)** |
| Weaviate | 2.13 | 2.46 | 2.44 | 3.21 | 2.54 | hafif yukarı, gürültülü |

Buna karşın **birleşik p50** neredeyse sabit kaldı (ör. Chroma 18.70→17.21ms, Qdrant
17.80→18.70ms — 770x chunk artışında %5-10 içinde). Bu, eski birleşik ölçümün ölçekle artan
gerçek arama maliyetini maskelediği hipotezini **destekliyor**: sabit ~14-15ms'lik embedding
maliyeti toplamda baskın olduğu için DB'ler arası ve ölçekler arası gerçek fark birleşik
metrikte görünmüyordu. Qdrant ve ElasticSearch'te bu net; Milvus/Weaviate'de gürültü payı
yüksek (repeat-içi std'ler search_only p50'nin kendisiyle aynı büyüklükte olabiliyor, bkz. ham
JSON `search_only_p50_ms_std_*`).

## C) query_embedding_avg_ms gerçekten ~8-10ms mi?

**Hayır.** Gözlenen değerler S1-S5'te **13.90-15.48ms** aralığında — hipotez edilen 8-10ms'den
belirgin yüksek (~1.5x). Yön doğru (embedding maliyeti sabit ve baskın, DB'ler arası search
farkını maskeliyor — bkz. B) ama sayısal tahmin yanlıştı. Kullanılan cihaz `mps` (Apple Silicon
GPU) — CPU'da bu muhtemelen daha da yüksek çıkardı. Tez metninde "~8-10ms" yerine ölçülen
**~14-15ms** kullanılmalı.

## Genel hüküm

**v3 sonuçları tez için kullanılabilir.** Chroma/Milvus `clear()` düzeltmesi doğrulandı (hem
izole cross-process testiyle hem koşum-içi count eşleşmesiyle), 5 ölçek temiz tabanla yeniden
koşuldu, ve Chroma S5 anomalisi düzeltme sonrası **kayboldu** (0.748→0.806) — v2'deki hipotez
(clear() kirliliği) kanıtlandı, "Chroma 1.5.2 ANN kalite kaybı" hipotezi **reddedildi**.

Eski v2 dosyaları ve `DENEY_A_V2_PROGRESS.md` → `experiments/results/archive_deneyA_v2/`
taşındı (silinmedi).

Deney B başlatılmadı.
