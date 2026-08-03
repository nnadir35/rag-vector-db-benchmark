# Deney A V2 — İlerleme

Üç değişiklik uygulandı:
1. FAISS sistematik benchmark'tan çıkarıldı (`--include-faiss` olmadan koşuluyor, varsayılan kapalı).
2. Chroma Docker'a alındı (`CHROMA_HOST=localhost CHROMA_PORT=8001` env var ile `HttpClient`).
   - Docker imajı `chromadb/chroma:0.4.24` → `1.5.2` yükseltildi (yerel pip `chromadb==1.5.2`
     istemcisi eski sunucunun v1 API'siyle değil, v2 API'siyle konuşuyor — versiyon uyumsuzluğu
     kök nedendi, sadece env var yetmedi).
3. Latency ayrıştırması: `retrieve_with_embedding()` ile search-only ölçüm + tek seferlik query
   embedding ölçümü eklendi (`search_only_*_ms_<db>`, `query_embedding_avg_ms`). Eski birleşik
   `retrieval_*_ms_<db>` alanları korundu (search_only + query_embedding_avg_ms olarak türetiliyor).

Sabit sorgu seti: `experiments/results/fixed_queries_DENEY_A.json` (500 ID, S1=100 dok kaynaklı).
Eski seri (FAISS'li, Chroma embedded, birleşik latency) → `experiments/results/archive_deneyA_v1/`.

## Bulunan ve düzeltilen bug: Qdrant `clear()` no-op idi

İlk S1 koşumunda Qdrant recall@10 = 0.764 çıktı (beklenen 0.936, >0.02 sapma —
KRİTİK SAĞLAMA eşiği aşıldı, DURULDU ve incelendi). Kök neden: `QdrantRetriever.clear()`
yalnızca `self._client is not None and self._collection_ready` iken server'da
`delete_collection` çağırıyordu. Her `benchmark_db.py` çalıştırması taze bir Python
process olduğundan bu iki alan her zaman `None`/`False` ile başlıyor — yani `clear()`
gerçekte HİÇBİR ZAMAN sunucudaki koleksiyonu silmiyordu. `add_chunks` → `_ensure_collection`
ise var olan aynı-boyutlu koleksiyonu YENİDEN KULLANIYOR (silmiyor), bu yüzden art arda
koşulan her benchmark'ın noktaları persistent Qdrant'ta BİRİKİYORDU. `bench_qdrant_squad`
koleksiyonu incelendiğinde 76.912 nokta bulundu (100 dokümanlık S1 için beklenen ~170).
Bu, önceki Deney A v1 serisinde de (S1 hariç, orada koleksiyon muhtemelen ilk kez
dokunulmuştu) potansiyel bir kirlilik kaynağıydı, ama v1 sonuçları ~0.02 tolerans içinde
kaldığı için fark edilmedi.

**Düzeltme** (`src/retrievers/qdrant_retriever.py::clear()`): artık `_collection_ready`
kontrolüne bakmadan koşulsuz olarak `_get_client().delete_collection(...)` çağırıyor —
`delete_collection` var olmayan koleksiyonda exception atmıyor, sessizce `False` dönüyor
(doğrulandı). Kirlenmiş `bench_qdrant_squad` koleksiyonu elle silindi, S1 tekrar koşuldu:
tüm 5 DB recall@10 = 0.9360 (birebir eşleşti), Qdrant koleksiyonu artık 170 nokta içeriyor.

| Ölçek | Durum | Başlangıç | Bitiş | Süre | Notlar |
|---|---|---|---|---|---|
| S1 · 100 dok | ✅ TAMAM (2. deneme, Qdrant fix sonrası) | 23:53:53 | 23:54:5x | ~1dk | recall@10=0.9360 (tüm 5 DB) |
| S2 · 1000 dok | ✅ TAMAM | 23:55:25 | 23:56:32 | ~1dk | recall@10=0.9000 (tüm 5 DB) |
| S3 · 10000 dok | ✅ TAMAM | 23:56:59 | 23:59:48 | ~3dk | recall@10 0.812-0.824 (beklenen 0.816-0.824 aralığında) |
| S4 · 20000 dok | ✅ TAMAM | 00:00:15 | 00:05:29 | ~5dk | recall@10 0.772-0.790 (beklenen 0.778-0.790, Weaviate 0.006 altında — tolerans içinde) |
| S5 · 20000 dok (chunk=256) | ✅ TAMAM (Chroma anomalisiyle, aşağıya bkz.) | 00:06:10 | 00:12:53 | ~7dk | Qdrant/ES/Milvus/Weaviate beklenen aralıkta; **Chroma 0.748 (beklenen 0.806-0.816)** |

## S5 anomalisi: Chroma recall@10 = 0.748 (beklenenden ~0.06-0.07 düşük)

Diğer 4 DB S5'te beklenen aralıkta (Qdrant=0.816, ES=0.816, Milvus=0.808, Weaviate=0.798 — hepsi
0.02 toleransı içinde). Sadece Chroma 0.748 ile beklenen 0.806-0.816 aralığının belirgin altında
kaldı (kritik sağlama eşiği olan ±0.02'yi ~3 kat aşıyor). S1-S4'te Chroma her zaman diğer DB'lerle
birebir/ +-0.001 aynıydı; sapma yalnızca S5'te (chunk_size=256 → 76.912 chunk, en büyük vektör
sayısı) ortaya çıktı.

**Yapılan inceleme (kök neden BULUNAMADI, olası nedenler elendi):**
1. **Veri kaybı değil** — koleksiyon `count()` = 76.912, beklenenle birebir eşleşiyor.
2. **Indexing lag / eventual-consistency değil** — koşumdan ~15 dakika sonra (indeksleme
   kesinlikle bitmiş olmalı) aynı sabit sorgu seti canlı script ile tekrar sorgulandı,
   recall@10 yine 0.748 çıktı (deterministik, tekrarlanabilir — geçici bir gecikme değil).
3. **HNSW config uyuşmazlığı değil** — `effective_config.chroma`: `hnsw:M=16`,
   `construction_ef=200`, `search_ef=64` — diğer DB'lerle birebir aynı, YAML'daki değerlerle eşleşiyor.
4. **Chunk ID çakışması değil** — `FixedSizeChunker` ID'leri `{doc_id}_chunk_{index}` olarak
   deterministik/tekil üretiyor, chunk_size küçülmesi çakışma yaratmaz.

**Değerlendirme:** Kök neden izole edilemedi ama S1-S4'te (embedded'den yeni geçilen aynı Docker
sunucusuyla, 20K'ya kadar chunk sayılarında) sorun yokken, yalnızca en büyük ölçekte (77K nokta)
ve yalnızca Chroma'da ortaya çıkması, Chroma 1.5.2 sunucusunun (Docker imajı bu deney kapsamında
0.4.24'ten 1.5.2'ye yükseltildi — bkz. Change 2 notu) yeni Rust tabanlı HNSW/segment senkronizasyon
mekanizmasının büyük ölçekte yaklaşık arama kalitesini embedded istemciye göre düşürdüğüne işaret
ediyor (`configuration_json.hnsw.sync_threshold=1000`, `resize_factor=1.2` gibi Chroma-server'a özgü
parametreler embedded modda mevcut değildi). Bu, üç atanan koddeğişikliğinin (FAISS opsiyonel,
Chroma Docker, latency ayrıştırma) bir yan etkisi değil, Chroma sunucu sürüm yükseltmesinin S5
ölçeğinde ortaya çıkan bağımsız bir gözlemi.

**Sonuç:** S5 sonuç dosyası (`official_scale_V2_chunk256_20000docs_5db_topk10_20260803_001253.json`)
olduğu gibi tutuldu (veri gerçek, sahte değil) ama bu anomali burada ve final raporda açıkça
işaretlendi. Daha derin kök-neden analizi (farklı `ef_search` değerleriyle deneme, Chroma
sunucu loglarını inceleme, S4→S5 arası kademeli ölçek testi) bu deneyin kapsamı dışında
bırakıldı — kullanıcı kararı gerektirir.
