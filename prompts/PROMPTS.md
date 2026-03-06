# PROMPTS.md — Hazır Prompt Şablonları

Her modül için kullanmaya hazır prompt'lar.
Claude veya Cursor'a **olduğu gibi kopyala-yapıştır** yapabilirsin.
Köşeli parantez içindeki kısımları değiştir: [BÖYLE]

---

## 0. Her Sohbetin Başına Ekle (Zorunlu Prefix)

```
AGENTS.md ve src/core/ içindeki tüm dosyaları oku.
Bu proje bir RAG benchmark framework'ü.
Mevcut kod stiline ve mimarisine tam uygun kod üret.
src/retrievers/pinecone_retriever.py referans implementasyon olarak kullan.
```

---

## 1. FixedSizeChunker

```
AGENTS.md'yi oku.

GÖREV: src/chunkers/ klasörüne FixedSizeChunker implementasyonu yaz.

Yapılacaklar:
1. src/chunkers/config.py → FixedSizeChunkerConfig dataclass
   - chunk_size: int (default=512)
   - overlap: int (default=50)
   - Validation: overlap < chunk_size olmalı

2. src/chunkers/fixed_size_chunker.py → FixedSizeChunker sınıfı
   - src/core/chunking.py'deki Chunker ABC'yi implemente et
   - chunk() methodu: Document alır, Sequence[Chunk] döner
   - Chunk ID formatı: "{document_id}_chunk_{chunk_index}" olsun
   - start_char ve end_char ChunkMetadata'ya doğru yazılsın
   - Overlap şöyle çalışsın: her chunk, öncekinin son overlap kadar karakterini de içersin

3. src/chunkers/__init__.py → export'ları yaz

4. tests/chunkers/test_fixed_size_chunker.py → testler
   - test_chunk_splits_document_into_correct_sizes
   - test_chunk_overlap_is_applied_correctly
   - test_chunk_ids_are_unique_and_deterministic
   - test_chunk_metadata_has_correct_positions
   - test_empty_document_returns_empty_list
   - test_document_shorter_than_chunk_size_returns_single_chunk

Kısıtlar:
- Hardcoded değer yok, her şey config'den gelmeli
- Tüm type hint'ler tam olmalı
- src/core/types.py'deki Document ve Chunk tiplerini kullan
```

---

## 2. SentenceTransformersEmbedder

```
AGENTS.md'yi oku.

GÖREV: src/embedders/ klasörüne SentenceTransformersEmbedder implementasyonu yaz.

Yapılacaklar:
1. src/embedders/config.py → SentenceTransformersEmbedderConfig dataclass
   - model_name: str (default="all-MiniLM-L6-v2")
   - device: str (default="cpu")
   - batch_size: int (default=32)
   - normalize_embeddings: bool (default=True)

2. src/embedders/sentence_transformers_embedder.py → SentenceTransformersEmbedder sınıfı
   - src/core/embedding.py'deki Embedder ABC'yi implemente et
   - embed_chunk(), embed_chunks(), embed_query(), get_dimension() methodları
   - embed_chunks() batch işlem yapsın (döngüyle değil, model.encode() ile)
   - Dimension, modelden otomatik alınsın (config'e yazılmasın)
   - Lazy loading: model __init__'te değil, ilk kullanımda yüklensin

3. src/embedders/__init__.py → export'ları yaz

4. tests/embedders/test_sentence_transformers_embedder.py → testler
   - Gerçek model yükleme testleri @pytest.mark.integration ile işaretle
   - Unit testlerde sentence_transformers mock'la
   - test_embed_chunk_returns_correct_dimension
   - test_embed_chunks_batch_matches_individual_results
   - test_get_dimension_is_consistent

Kısıtlar:
- sentence-transformers paketi lazy import edilmeli (installed değilse ImportError ver)
- Embedding vektörleri Embedding(vector=..., dimension=...) olarak dön
- src/retrievers/pinecone_retriever.py'deki lazy import pattern'ını takip et
```

---

## 3. OpenAIGenerator

```
AGENTS.md'yi oku.

GÖREV: src/generators/ klasörüne OpenAIGenerator implementasyonu yaz.

Yapılacaklar:
1. src/generators/config.py → OpenAIGeneratorConfig dataclass
   - model_name: str (default="gpt-4o-mini")
   - temperature: float (default=0.0)  ← 0.0 çünkü reproducibility önemli
   - max_tokens: int (default=512)
   - system_prompt: str (default aşağıda)
   - context_template: str (nasıl formatlanacağı)
   - api_key: Optional[str] (env'den de okunabilmeli: OPENAI_API_KEY)

2. src/generators/openai_generator.py → OpenAIGenerator sınıfı
   - src/core/generation.py'deki Generator ABC'yi implemente et
   - generate(query, retrieved_chunks) → GenerationResult
   - retrieved_chunks'ları prompt'a şöyle ekle:
     "Context:\n[1] {chunk1.content}\n[2] {chunk2.content}\n\nQuestion: {query.text}"
   - metadata'ya şunları ekle: model_name, latency_seconds, prompt_tokens, completion_tokens, total_cost_usd
   - Cost hesaplama: gpt-4o-mini için input $0.15/1M token, output $0.60/1M token

3. src/generators/__init__.py → export'ları yaz

4. tests/generators/test_openai_generator.py → testler
   - openai.OpenAI client'ını mock'la (gerçek API çağrısı yapma)
   - test_generate_formats_prompt_correctly
   - test_generate_returns_generation_result_with_metadata
   - test_generate_includes_latency_in_metadata
   - test_api_key_read_from_environment

Default system prompt:
"You are a helpful assistant. Answer the question based only on the provided context. 
If the answer is not in the context, say 'I don't know'."
```

---

## 4. RetrievalEvaluator

```
AGENTS.md'yi oku.

GÖREV: src/evaluators/ klasörüne RetrievalEvaluator implementasyonu yaz.

Gerekli Bilgi — Bu metrikleri implemente et:
- Precision@K: Getirilen K sonuç içinde kaçı gerçekten alakalı? (alakalı_bulunan / K)
- Recall@K: Tüm alakalı dokümanların kaçı K içinde bulundu? (alakalı_bulunan / toplam_alakalı)
- MRR (Mean Reciprocal Rank): İlk doğru sonucun sırasının tersi (1/rank)
- nDCG@K: Sıralama kalitesi — üstteki sonuçlar daha çok puan alır

Yapılacaklar:
1. src/evaluators/config.py → RetrievalEvaluatorConfig dataclass
   - k_values: List[int] (default=[1, 3, 5, 10])

2. src/evaluators/metrics.py → Pure fonksiyonlar olarak metrik hesaplamaları
   - precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) → float
   - recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) → float
   - mrr(retrieved_ids: List[str], relevant_ids: Set[str]) → float
   - ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) → float
   - Bu fonksiyonlar PURE olmalı (side effect yok, sadece hesaplama)

3. src/evaluators/retrieval_evaluator.py → RetrievalEvaluator sınıfı
   - evaluate(retrieval_result: RetrievalResult, ground_truth: Set[str]) → Dict[str, float]
   - Tüm k_values için tüm metrikleri hesapla
   - Dönen dict örneği: {"precision@1": 1.0, "precision@3": 0.67, "mrr": 1.0, ...}

4. src/evaluators/__init__.py → export'ları yaz

5. tests/evaluators/test_metrics.py → her metrik için birim testler
   - Örnek: precision@3 = 2/3 olduğunda doğru sonuç
   - Edge case: boş retrieved list → 0.0
   - Edge case: tüm sonuçlar alakalı → 1.0
```

---

## 5. GenerationEvaluator (LLM-as-judge)

```
AGENTS.md'yi oku.

GÖREV: src/evaluators/generation_evaluator.py yaz.

Bu evaluator LLM'i "hakem" olarak kullanır (LLM-as-a-judge pattern).
Üretilen cevabın kalitesini başka bir LLM değerlendirir.

Ölçülecek metrikler:
- faithfulness: Cevap sadece verilen context'ten mi geliyor? (0-1)
- answer_relevancy: Cevap soruyla ne kadar alakalı? (0-1)
- context_precision: Context içindeki bilginin ne kadarı kullanıldı? (0-1)

Yapılacaklar:
1. src/evaluators/generation_evaluator.py → GenerationEvaluator sınıfı
   - evaluate(rag_response: RAGResponse, ground_truth_answer: Optional[str]) → Dict[str, float]
   - Faithfulness için judge prompt'u config'den gelmeli
   - LLM çağrısı için Generator interface'ini kullan (hardcode openai değil)

2. src/evaluators/judge_prompts.py → Hakem prompt'larını bir yerde topla
   - FAITHFULNESS_PROMPT: str
   - ANSWER_RELEVANCY_PROMPT: str
   - JSON formatında skor döndürmesini iste (parse edilebilsin diye)

3. tests/evaluators/test_generation_evaluator.py → testler
   - judge LLM'i mock'la
   - test_evaluate_returns_scores_between_0_and_1
   - test_faithfulness_prompt_includes_context_and_answer
```

---

## 6. RAGPipeline

```
AGENTS.md'yi oku.

GÖREV: src/pipeline/rag_pipeline.py yaz.

Pipeline şu sırayı takip eder:
1. query gelir
2. retriever.retrieve(query) → RetrievalResult
3. generator.generate(query, retrieval_result.chunks) → GenerationResult
4. (opsiyonel) retrieval_evaluator.evaluate(retrieval_result, ground_truth)
5. (opsiyonel) generation_evaluator.evaluate(rag_response)
6. RAGResponse + metrics döner

Yapılacaklar:
1. src/pipeline/config.py → RAGPipelineConfig
   - retriever_name: str
   - generator_name: str
   - top_k: int (default=5)
   - evaluate_retrieval: bool (default=True)
   - evaluate_generation: bool (default=False)  ← pahalı olduğu için default False

2. src/pipeline/rag_pipeline.py → RAGPipeline sınıfı
   - __init__(retriever, generator, retrieval_evaluator=None, generation_evaluator=None)
   - run(query: Query, ground_truth_ids: Optional[Set[str]]) → PipelineResult
   - run_batch(queries, ground_truths) → List[PipelineResult]
   - PipelineResult: query + rag_response + metrics + total_latency_seconds

3. src/pipeline/__init__.py → export'lar

4. tests/pipeline/test_rag_pipeline.py → testler
   - retriever ve generator mock'la
   - test_run_returns_pipeline_result_with_response
   - test_run_includes_retrieval_metrics_when_evaluator_provided
   - test_run_batch_processes_all_queries
```

---

## 7. SQuAD Dataset Loader

```
AGENTS.md'yi oku.

GÖREV: src/datasets/squad_loader.py yaz.

SQuAD (Stanford Question Answering Dataset) formatını yükler.
HuggingFace datasets kütüphanesi kullan: from datasets import load_dataset

Yapılacaklar:
1. src/datasets/config.py → SQuADDatasetConfig
   - split: str (default="validation")
   - max_samples: Optional[int] (test için küçük subset almak için)
   - version: str (default="squad_v2")

2. src/datasets/squad_loader.py → SQuADLoader
   - load() → Tuple[List[Query], Dict[str, Set[str]]]
     - List[Query]: sorular
     - Dict[str, Set[str]]: query_id → alakalı context id'leri (ground truth)
   - load_documents() → List[Document]: context paragrafları Document olarak

3. src/datasets/__init__.py → export'lar

Not: SQuAD'da her soru bir context paragrafına ait.
Ground truth = o sorunun context_id'si.
Yani ideal retriever, soruyu sorduğunda o context'i en üste getirmeli.
```

---

## 8. İlk Deney Config'i

```
AGENTS.md'yi oku.

GÖREV: experiments/configs/baseline_pinecone_openai.yaml dosyasını oluştur.

Bu config şunu tanımlar:
- Hangi retriever (pinecone), hangi embedder (sentence-transformers)
- Hangi generator (openai gpt-4o-mini)
- Hangi dataset (squad_v2, validation, max 100 sample)
- Hangi metrikler (precision@1, precision@5, recall@5, mrr, ndcg@5)
- Hangi k değerleri (1, 3, 5, 10)

Ayrıca experiments/configs/baseline_chroma_openai.yaml oluştur:
Aynı şey ama retriever = chroma (local, ücretsiz)

Bu iki config sayesinde Pinecone vs Chroma karşılaştırması yapabileceğiz.
```

---

## 9. Chroma Retriever (2. Vektör DB)

```
AGENTS.md'yi oku.

GÖREV: src/retrievers/chroma_retriever.py yaz.

pinecone_retriever.py ile AYNI interface'i kullan (Retriever ABC).
chromadb paketini kullan (pip install chromadb).

Fark: Chroma local çalışır, API key gerekmez, persist_directory ile diske kaydeder.

Config:
- collection_name: str
- persist_directory: Optional[str] (None ise in-memory çalışır)
- distance_metric: str (default="cosine")
- dimension: int

PineconeRetriever ile birebir aynı method imzaları:
- add_chunks(chunks, embeddings)
- retrieve(query, top_k)
- retrieve_with_embedding(query_embedding, top_k, query_id)
- clear()

Bu sayede pipeline'da sadece config değişecek, kod değişmeyecek.
```

---

## 10. run_experiment.py Script'i

```
AGENTS.md'yi oku.

GÖREV: scripts/run_experiment.py CLI script'i yaz.

Kullanım: python scripts/run_experiment.py --config experiments/configs/baseline_pinecone_openai.yaml

Yaptıkları:
1. Config YAML'ı yükle
2. Component'leri registry'den instantiate et
3. Dataset'i yükle
4. Pipeline'ı çalıştır
5. Sonuçları experiments/results/{timestamp}_{experiment_name}.json olarak kaydet
6. Özet tabloyu terminale yazdır

Output JSON formatı:
{
  "experiment_name": "baseline_pinecone_openai",
  "timestamp": "2024-01-15T10:30:00",
  "config": {...},
  "metrics": {
    "precision@1": 0.72,
    "precision@5": 0.61,
    "recall@5": 0.85,
    "mrr": 0.78,
    "ndcg@5": 0.74,
    "avg_retrieval_latency_seconds": 0.23,
    "avg_generation_latency_seconds": 1.1
  },
  "per_query_results": [...]
}

argparse kullan. --config parametresi zorunlu.
```

---

## Genel Hata Düzeltme Promptu

```
Şu hata mesajını alıyorum:
[HATA MESAJINI BURAYA YAPISTIR]

İlgili dosya: [DOSYA YOLU]
İlgili kod:
[KODU BURAYA YAPISTIR]

AGENTS.md kurallarını ihlal etmeden düzelt.
Sadece sorunu çözen minimum değişikliği yap.
```

---

## Kod Review Promptu

```
Şu dosyayı yaz: [DOSYA YOLU]
[KODU BURAYA YAPISTIR]

Şu açılardan değerlendir:
1. AGENTS.md kurallarına uyuyor mu?
2. src/core/ interface'leriyle tam uyumlu mu?
3. Type hint eksik var mı?
4. Hardcoded değer var mı?
5. Test edilebilir mi (mock'lanabilir bağımlılıklar)?
Varsa düzelt, yoksa onay ver.
```
