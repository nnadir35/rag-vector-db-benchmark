# Deney A — İlerleme

| Ölçek | Durum | Başlangıç | Bitiş | Süre | Notlar |
|---|---|---|---|---|---|
| S1 · 100 dok · ~190 vektör  | ✅ TAMAM | 22:40:24 | 22:42:58 | 2dk34sn | recall@10=0.9360 (tüm DB'ler) |
| S2 · 1K dok · ~1.9K vektör  | ✅ TAMAM | 22:43:46 | 22:46:38 | 2dk52sn | recall@10≈0.90, Weaviate hafif düşük (0.898) |
| S3 · 10K dok · ~19K vektör  | ✅ TAMAM | 22:47:00 | 22:51:46 | 4dk46sn | recall@10 0.816-0.824 arası, ölçek büyüdükçe düşüyor |
| S4 · 20K dok · ~38K vektör  | ✅ TAMAM (fix sonrası) | 23:03:03 | 23:10:31 | 7dk28sn | recall@10 0.778-0.790 arası, Milvus batch_size=25 fix'i ile sorunsuz |
| S5 · 20K dok · ~70K vektör (chunk_size=256) | ✅ TAMAM | 23:11:13 | 23:20:10 | 8dk57sn | recall@10 0.806-0.816, S4'e (chunk=512) göre biraz daha yüksek |

**Toplam tahmini kalan süre:** 0 — tüm ölçekler tamamlandı
**Son güncelleme:** 23:20

## S5 Sonuçları (8dk57sn, chunk_size=256)
| DB | recall@10 | p50 ms | p95 ms | index toplam sn | DB yazma sn |
|---|---|---|---|---|---|
| Chroma | 0.8060 | 12.30 | 16.82 | 198.844 | 52.416 |
| Qdrant | 0.8160 | 17.36 | 21.70 | 174.111 | 27.684 |
| FAISS | 0.8060 | 10.64 | 14.61 | 175.668 | 29.241 |
| Milvus | 0.8087 | 15.79 | 24.08 | 168.857 | 22.430 |
| ES | 0.8160 | 18.99 | 28.56 | 202.115 | 55.688 |
| Weaviate | 0.8120 | 17.77 | 27.90 | 175.409 | 28.982 |

En hızlı (p50): FAISS 10.6ms | En yüksek MRR: Qdrant 0.637
effective_config uyarısı: YOK
embedding_seconds (S5, 20000 dok, chunk=256) = 146.427s
YAML `chunker.chunk_size` işlem sonrası 512'ye geri alındı (doğrulandı).
Sonuç dosyası `official_scale_20000docs_chunk256_6db_topk10_20260802_232007.json` olarak yeniden adlandırıldı.

## S4 ÇÖZÜM NOTU
`src/retrievers/milvus_retriever.py` `add_chunks` içinde tüm entity'ler tek `client.insert()` çağrısında
gönderiliyordu (hiç batching yoktu) — 20K dokümanda ~38K chunk'lık embedding payload'u Milvus gRPC
istemcisinin 64MB max mesaj limitini aşıyordu. Kullanıcı onayıyla `batch_size = 25` ile parçalı insert
eklendi (25'lik gruplar halinde `client.insert()` döngüsü). Fix sonrası S4 sorunsuz tamamlandı.

## S4 Sonuçları (7dk28sn, fix sonrası)
| DB | recall@10 | p50 ms | p95 ms | index toplam sn | DB yazma sn |
|---|---|---|---|---|---|
| Chroma | 0.7800 | 13.85 | 27.31 | 169.399 | 32.239 |
| Qdrant | 0.7900 | 17.25 | 26.08 | 152.872 | 15.713 |
| FAISS | 0.7840 | 10.74 | 15.93 | 150.946 | 13.787 |
| Milvus | 0.7840 | 21.82 | 42.41 | 150.758 | 13.598 |
| ES | 0.7900 | 20.03 | 33.55 | 163.966 | 26.807 |
| Weaviate | 0.7780 | 17.01 | 26.11 | 155.343 | 18.183 |

En hızlı (p50): FAISS 10.7ms | En yüksek MRR: ElasticSearch 0.601
effective_config uyarısı: YOK
embedding_seconds (S4, 20000 dok) = 137.160s

## S3 Sonuçları (4dk46sn)
| DB | recall@10 | p50 ms | p95 ms | index toplam sn | DB yazma sn |
|---|---|---|---|---|---|
| Chroma | 0.8200 | 11.91 | 17.10 | 79.931 | 12.099 |
| Qdrant | 0.8240 | 15.89 | 20.67 | 74.701 | 6.869 |
| FAISS | 0.8160 | 10.75 | 16.04 | 72.021 | 4.190 |
| Milvus | 0.8167 | 16.41 | 27.77 | 76.001 | 8.169 |
| ES | 0.8240 | 17.30 | 24.78 | 78.878 | 11.047 |
| Weaviate | 0.8180 | 16.91 | 26.40 | 76.173 | 8.341 |

En hızlı (p50): FAISS 10.7ms | En yüksek MRR: Qdrant 0.633
effective_config uyarısı: YOK
embedding_seconds (S3, 10000 dok) = 67.832s
**Not:** Süre tahmin formülü gerçek veriyle revize edildi — overhead sabit değil, DB yazma/indeksleme de ölçekle artıyor ama toplam koşum süresi ilk tahminden çok daha kısa çıktı (4dk46sn vs tahmini ~19dk). S4/S5 tahminleri buna göre aşağı çekildi.

## S2 Sonuçları (2dk52sn)
| DB | recall@10 | p50 ms | p95 ms | index toplam sn | DB yazma sn |
|---|---|---|---|---|---|
| Chroma | 0.9000 | 11.66 | 15.59 | 15.745 | 1.573 |
| Qdrant | 0.9000 | 15.82 | 20.47 | 15.643 | 1.471 |
| FAISS | 0.9000 | 10.39 | 14.57 | 14.436 | 0.264 |
| Milvus | 0.9000 | 15.70 | 20.66 | 20.628 | 6.456 |
| ES | 0.9000 | 16.80 | 23.45 | 15.709 | 1.537 |
| Weaviate | 0.8980 | 16.06 | 20.94 | 16.051 | 1.879 |

En hızlı (p50): FAISS 10.4ms | En yüksek MRR: ChromaDB 0.715
effective_config uyarısı: YOK (sadece zararsız ResourceWarning/DeprecationWarning'ler)
embedding_seconds (S2, 1000 dok) = 14.172s

## S1 Sonuçları (2dk34sn)
| DB | recall@10 | p50 ms | p95 ms | index toplam sn | DB yazma sn |
|---|---|---|---|---|---|
| Chroma | 0.9360 | 11.13 | 16.05 | 7.320 | 0.562 |
| Qdrant | 0.9360 | 15.41 | 19.78 | 7.440 | 0.683 |
| FAISS | 0.9360 | 10.22 | 13.89 | 6.771 | 0.013 |
| Milvus | 0.9360 | 14.96 | 19.15 | 11.835 | 5.078 |
| ES | 0.9360 | 16.53 | 21.50 | 7.065 | 0.307 |
| Weaviate | 0.9360 | 15.38 | 19.35 | 8.068 | 1.310 |

En hızlı (p50): FAISS 10.2ms | En yüksek MRR: ChromaDB 0.766 (tüm DB'lerde MRR/nDCG aynı: 0.7658 / 0.8075)
effective_config uyarısı: YOK
embedding_seconds (S1, 100 dok) = 6.757s → 0.06757 s/dok

**Süre tahmin formülü (embedding hızına dayalı, overhead dahil):**
- S2 tahmini = 6.757×10 + ~150s overhead ≈ 218s (~3.6dk)
- S3 tahmini = 6.757×100 + ~450s overhead ≈ 1125s (~18.8dk)
- S4 tahmini = 6.757×200 + ~750s overhead ≈ 2101s (~35dk)
- S5 tahmini = S4×1.3 ≈ 2731s (~45.5dk)

## S4 HATASI (✅ ÇÖZÜLDÜ — bkz. "S4 ÇÖZÜM NOTU" yukarıda)
`Milvus` add_chunks çağrısında crash:
```
grpc._channel._InactiveRpcError: RESOURCE_EXHAUSTED
grpc: received message larger than max (83120441 vs. 67108864)
```
20.000 dokümandan üretilen chunk+embedding batch'i (100 chunk'lık toplu insert) Milvus gRPC istemcisinin
varsayılan max mesaj boyutunu (64MB) aşıyor (83MB). `scripts/benchmark_db.py` bu hatayı tek bir DB için
izole etmiyor — exception `main()`'e kadar yükseliyor ve **tüm script'i durduruyor**, bu yüzden S4 için
6 DB'nin HİÇBİRİNE ait sonuç dosyası üretilemedi (Milvus dahil).

Talimattaki kural #3 ("Bir DB hata verirse: o DB için null yaz, diğer 5 ile devam et") şu anki kod ile
uygulanamıyor çünkü izolasyon yok — bu bir kod değişikliği gerektiriyor (ya Milvus batch_size'ı chunk
sayısına göre küçültmek, ya da run_benchmark() içinde her DB bloğunu try/except ile sarmalamak).
Bu, otonom karar kurallarının kapsamı dışında bir kod değişikliği olduğu için burada DURULDU.

**Olası çözümler (kullanıcı onayı gerekir):**
1. `milvus_retriever.py`'de `add_chunks` içindeki batch_size'ı (100) daha küçük bir değere düşürmek
   (örn. 20-30) — 20K dokümanda ~38K chunk × embedding boyutu 100'lük batch'lerde 64MB'ı aşıyor.
2. Milvus istemci tarafında `grpc.max_receive_message_length` gibi bir ayarla max mesaj boyutunu artırmak.
3. `run_benchmark()` içinde her DB bloğunu try/except RuntimeError ile sarmalayıp diğer DB'lerin
   etkilenmemesini sağlamak (kalıcı kod iyileştirmesi, CLAUDE.md "Adding a New Retriever" desenine uygun
   şekilde ileride eklenebilir).

**Durum:** S1-S3 tamamlandı ve sağlıklı. S4-S5 kullanıcı kararı bekliyor.

## Ortam notları
- Docker container'lar kontrol edildi: qdrant_db, milvus_db, elasticsearch_db, weaviate, chroma_db hepsi ayakta.
- chroma_db "unhealthy" görünüyor ama sebep healthcheck script'inin container içinde `curl` bulamaması — servis kendisi çalışıyor (host portu 8001, `/api/v2/heartbeat` yanıt veriyor). Fonksiyonel bir engel değil, göz ardı edildi.
- fixed_queries_20260802_224257.json → experiments/results/fixed_queries_DENEY_A.json olarak kopyalandı, S2-S5'te bu kullanılacak.

---

### DENEY A TAMAMLANDI

| Ölçek | ~Vektör | En yüksek recall | En düşük recall | En hızlı DB (p50) |
|---|---|---|---|---|
| S1 (100 dok) | ~190 | 0.9360 (tüm DB'ler eşit) | 0.9360 | FAISS 10.2ms |
| S2 (1K dok) | ~1.9K | 0.9000 (Chroma/Qdrant/FAISS/Milvus/ES) | 0.8980 (Weaviate) | FAISS 10.4ms |
| S3 (10K dok) | ~19K | 0.8240 (Qdrant/ES) | 0.8160 (FAISS) | FAISS 10.7ms |
| S4 (20K dok, chunk=512) | ~38K | 0.7900 (Qdrant/ES) | 0.7780 (Weaviate) | FAISS 10.7ms |
| S5 (20K dok, chunk=256) | ~70K | 0.8160 (Qdrant/ES) | 0.8060 (Chroma/FAISS) | FAISS 10.6ms |

**Gözlem:** Ölçek büyüdükçe (S1→S4) mutlak recall@10 düzenli biçimde düşüyor (0.936 → 0.778-0.79), ama
DB'ler arasındaki *fark* aslında açılmıyor — her ölçekte en iyi ile en kötü DB arasındaki recall farkı
hep dar kalıyor (~0.002-0.02 aralığında), yani zorluk artışı tüm DB'leri (aynı embedding/HNSW config
kullandıkları için) neredeyse eşit etkiliyor. FAISS tutarlı biçimde en düşük p50 latency'yi veriyor,
ama recall sıralamasında öne çıkan tek DB yok. S5'te chunk_size'ı 256'ya düşürmek recall'ı S4'e göre
belirgin şekilde artırdı (~0.78-0.79 → ~0.81-0.82) — daha küçük, daha odaklı chunk'lar bu ölçekte
retrieval kalitesini iyileştiriyor.

**Dosya bütünlüğü kontrolü:** Deney A'ya ait 5 official_scale_* dosyası da `experiments/results/`'da mevcut:
- `official_scale_100docs_6db_topk10_20260802_224257.json` (S1)
- `official_scale_1000docs_6db_topk10_20260802_224637.json` (S2)
- `official_scale_10000docs_6db_topk10_20260802_225145.json` (S3)
- `official_scale_20000docs_6db_topk10_20260802_231027.json` (S4, chunk=512, Milvus fix sonrası)
- `official_scale_20000docs_chunk256_6db_topk10_20260802_232007.json` (S5, chunk=256)

Eksik yok. (17:43/17:44 tarihli `100docs`/`1000docs` dosyaları bu oturumdan önceki eski koşumlara ait,
Deney A'nın parçası değil.)

**Deney B için not — FAISS flat modu / ground truth üretimi:**
`src/retrievers/config.py`'deki `FAISSRetrieverConfig`'te `index_type` alanı zaten var ve muhtemelen
`"flat"` ve `"hnsw"` seçeneklerini destekliyor (S1-S5'te `"hnsw"` kullanıldı). Deney B'de tam/doğru (exact)
en-yakın-komşu ground truth üretmek için `faiss_retriever.py`'de `index_type: "flat"` ile ayrı bir
FAISS index kurup tüm sorgular için brute-force top-k hesaplanması gerekecek — bu, HNSW'nin approximate
sonuçlarına karşı recall@10'un neye göre ölçüldüğünün referans noktası olacak. Şu an her ölçekte
recall@10 hesaplaması muhtemelen SQuAD'ın kendi ground-truth etiketlerine karşı yapılıyor (dataset-level),
flat-FAISS tabanlı bir ANN-vs-exact karşılaştırması değil — Deney B'nin bunu netleştirmesi gerekecek.
Bu inceleme yapılmadı, sadece config'te `index_type` alanının var olduğu doğrulandı; detaylı kod okuması
Deney B başında yapılmalı.

**DENEY B BAŞLATILMADI.**
