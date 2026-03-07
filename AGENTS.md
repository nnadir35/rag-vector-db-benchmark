# AGENTS.md
# RAG Vector DB Benchmark — Proje Hafızası
#
# ⚠️  BU DOSYAYI HER SOHBETIN BAŞINDA OKU
# ⚠️  HER MODÜL BİTİNCE DURUMU GÜNCELLE (✅ yap)
# ⚠️  ASLA mimariye aykırı kod yazma — kurallar aşağıda

---

## 📍 ŞU ANKİ DURUM

```
SON GÜNCELLEME : 07.03.2026
AKTİF GÖREV    : BİTTİ! Tüm Sistem Hazır! 🚀
SONRAKİ GÖREV  : Entegrasyon / Kullanıcı Testi
TAMAMLANAN     : 10/10
```

---

## ✅ TAMAMLANAN MODÜLLER

> Bir modülü bitirince buraya taşı ve tarihi yaz.

<!-- Örnek:
- [x] FixedSizeChunker — 2024-01-15 ✅
-->

- [x] 1. FixedSizeChunker — 07.03.2026 ✅
- [x] 2. SentenceTransformersEmbedder — 07.03.2026 ✅
- [x] 3. SQuAD Dataset Loader — 07.03.2026 ✅
- [x] 4. UniversalGenerator — 07.03.2026 ✅
- [x] 5. RetrievalEvaluator — 07.03.2026 ✅
- [x] 6. RAGPipeline — 07.03.2026 ✅
- [x] 7. config_loader.py — 07.03.2026 ✅
- [x] 8. run_experiment.py — 07.03.2026 ✅
- [x] 9. ChromaRetriever — 07.03.2026 ✅
- [x] 10. GenerationEvaluator — 07.03.2026 ✅

---

## 📋 YAPILACAKLAR (Sırayla)

Sırayı bozma. Her biri bir öncekine bağlı.

- [x] **1. FixedSizeChunker**
- [x] **2. SentenceTransformersEmbedder**
- [x] **3. SQuAD Dataset Loader**
- [x] **4. UniversalGenerator**
- [x] **5. RetrievalEvaluator + metrics.py**
- [x] **6. RAGPipeline**
- [x] **7. config_loader.py** 
- [x] **8. run_experiment.py**
- [x] **9. ChromaRetriever**
- [x] **10. GenerationEvaluator**

---

## 🚀 AKTİF GÖREV — Şu An Yapılacak

### GÖREV 1: FixedSizeChunker

**Ne üretilecek:**
```
src/chunkers/__init__.py
src/chunkers/config.py
src/chunkers/fixed_size_chunker.py
tests/chunkers/__init__.py
tests/chunkers/test_fixed_size_chunker.py
```

**Bitti mi kontrol:**
```bash
python -m pytest tests/chunkers/ -v
```
Tüm testler yeşilse → bu dosyada `- [ ]` → `- [x]` yap, SONRAKI GÖREV'i güncelle.

**PROMPT — Antigravity'e olduğu gibi yapıştır:**

```
Bu proje dosyasını (AGENTS.md) ve şu dosyaları oku:
- src/core/chunking.py       (implement edeceğin interface)
- src/core/types.py          (kullanacağın tipler)
- src/retrievers/pinecone_retriever.py  (kod stili referansı)
- src/retrievers/config.py   (config pattern referansı)

GÖREV: src/chunkers/ klasörüne FixedSizeChunker implementasyonu yaz.

1. src/chunkers/config.py → FixedSizeChunkerConfig dataclass
   - @dataclass(frozen=True)
   - chunk_size: int = 512
   - overlap: int = 50
   - __post_init__: overlap < chunk_size olduğunu doğrula

2. src/chunkers/fixed_size_chunker.py → FixedSizeChunker sınıfı
   - src/core/chunking.py içindeki Chunker ABC'yi tam implemente et
   - chunk(document) → Sequence[Chunk]
   - Chunk ID formatı: "{document_id}_chunk_{chunk_index}"
   - start_char / end_char her Chunk'ın ChunkMetadata'sına doğru yazılsın
   - Overlap: her chunk, öncekinin son `overlap` karakterini de içersin
   - Boş document → boş liste dön
   - document.content < chunk_size → tek chunk dön

3. src/chunkers/__init__.py → FixedSizeChunker ve FixedSizeChunkerConfig export et

4. tests/chunkers/__init__.py → boş dosya

5. tests/chunkers/test_fixed_size_chunker.py → şu testleri yaz:
   - test_chunk_splits_document_into_correct_number_of_chunks
   - test_chunk_overlap_is_applied_correctly
   - test_chunk_ids_are_unique_and_deterministic
   - test_chunk_metadata_has_correct_start_and_end_chars
   - test_empty_document_returns_empty_list
   - test_document_shorter_than_chunk_size_returns_single_chunk
   - test_config_raises_when_overlap_exceeds_chunk_size

KURALLAR:
- Tüm type hint'ler tam yazılacak
- Her public method'da docstring olacak
- Hardcoded değer olmayacak, her şey config'den gelecek
- pinecone_retriever.py'deki kod stilini takip et
```

---

## ⏭️ SONRAKI GÖREV — Hazırlık

### GÖREV 2: SentenceTransformersEmbedder

> Görev 1 bitmeden başlama.

**Ne üretilecek:**
```
src/embedders/__init__.py
src/embedders/config.py
src/embedders/sentence_transformers_embedder.py
tests/embedders/__init__.py
tests/embedders/test_sentence_transformers_embedder.py
```

**Bitti mi kontrol:**
```bash
python -m pytest tests/embedders/ -v
```

**PROMPT — Antigravity'e olduğu gibi yapıştır:**

```
Bu proje dosyasını (AGENTS.md) ve şu dosyaları oku:
- src/core/embedding.py      (implement edeceğin interface)
- src/core/types.py          (kullanacağın tipler)
- src/chunkers/fixed_size_chunker.py  (tamamlanan referans implementasyon)
- src/retrievers/config.py   (config pattern referansı)

GÖREV: src/embedders/ klasörüne SentenceTransformersEmbedder implementasyonu yaz.

1. src/embedders/config.py → SentenceTransformersEmbedderConfig dataclass
   - @dataclass(frozen=True)
   - model_name: str = "all-MiniLM-L6-v2"
   - device: str = "cpu"
   - batch_size: int = 32
   - normalize_embeddings: bool = True

2. src/embedders/sentence_transformers_embedder.py → SentenceTransformersEmbedder
   - src/core/embedding.py içindeki Embedder ABC'yi tam implemente et
   - embed_chunk(), embed_chunks(), embed_query(), get_dimension() methodları
   - embed_chunks(): döngü değil, model.encode() ile batch işlem yap
   - dimension, modelden otomatik alınsın — config'e yazılmasın
   - Lazy loading: model __init__'te değil, ilk kullanımda yüklensin
   - sentence_transformers paketi lazy import: yüklü değilse ImportError ver

3. src/embedders/__init__.py → export'lar

4. tests/embedders/__init__.py → boş dosya

5. tests/embedders/test_sentence_transformers_embedder.py → testler
   - sentence_transformers modülünü mock'la (gerçek model indirme)
   - test_embed_chunk_returns_correct_dimension
   - test_embed_chunks_batch_returns_same_count_as_input
   - test_get_dimension_is_consistent_with_embeddings
   - test_model_is_lazy_loaded_on_first_use
   - test_import_error_when_sentence_transformers_not_installed

KURALLAR (aynı):
- Tüm type hint'ler tam
- Her public method'da docstring
- Hardcoded değer yok
```

---

## ⏳ BEKLEYEN GÖREVLER

### GÖREV 3: SQuAD Dataset Loader

**PROMPT:**
```
AGENTS.md'yi oku.

GÖREV: src/datasets/ klasörüne SQuAD dataset loader yaz.

1. src/datasets/config.py → SQuADDatasetConfig
   - split: str = "validation"
   - max_samples: Optional[int] = None
   - version: str = "squad_v2"

2. src/datasets/squad_loader.py → SQuADLoader
   - load() → Tuple[List[Query], Dict[str, Set[str]]]
     - List[Query]: her soru bir Query nesnesi
     - Dict[str, Set[str]]: query_id → {context_id} (ground truth)
   - load_documents() → List[Document]
     - Her context paragrafı bir Document nesnesi
   - HuggingFace datasets kütüphanesi kullan: from datasets import load_dataset
   - Document ID: "squad_{context_hash}" formatında deterministik üret

3. src/datasets/__init__.py → export'lar
4. tests/datasets/test_squad_loader.py → testler (datasets mock'la)

NOT: SQuAD'da ground truth şudur: bir sorunun cevabı belirli bir context paragrafında.
İdeal retriever o soruyu query olarak verince o paragrafı en üste getirmeli.
```

---

### GÖREV 4: UniversalGenerator

**PROMPT:**
```
AGENTS.md'yi oku.

GÖREV: src/generators/ klasörüne LiteLLM tabanlı UniversalGenerator implementasyonu yaz.

1. src/generators/config.py → UniversalGeneratorConfig dataclass
   - @dataclass(frozen=True)
   - model_name: str = "ollama/llama3.1" (Varsayılan olarak yerel model)
   - temperature: float = 0.0
   - max_tokens: int = 512
   - api_base: Optional[str] = None (Ollama için http://localhost:11434 gerekebilir)
   - api_key: Optional[str] = None (Groq veya OpenAI için)
   - system_prompt: str = "Answer based only on the provided context."

2. src/generators/universal_generator.py → UniversalGenerator
   - src/core/generation.py içindeki Generator ABC'yi implemente et.
   - 'litellm' kütüphanesini kullan. Metot içinde 'from litellm import completion' şeklinde lazy import yap.
   - generate(query, retrieved_chunks) metodunda:
     - Context'i şablonla birleştir: "Context:\n{chunks}\n\nQuestion: {query}"
     - litellm.completion(model=self.config.model_name, messages=..., temperature=...) çağrısını yap.
     - Dönen cevabı (GenerationResult) oluştururken metadata'ya 'provider' bilgisini model isminden ayıklayıp ekle.

3. src/generators/registry.py → UniversalGenerator'ı register et.

4. tests/generators/test_universal_generator.py → Testleri yaz.
   - 'litellm.completion' fonksiyonunu mock'la.
   - test_ollama_provider_call (model_name="ollama/llama3" iken doğru çağrılıyor mu?)
   - test_groq_provider_call (model_name="groq/llama-3.1-8b" iken doğru çağrılıyor mu?)

KURALLAR:
- litellm kütüphanesi yüklü değilse ImportError verip kullanıcıyı uyarmalı.
- Response içindeki token kullanım (usage) bilgilerini mutlaka GenerationResult.metadata'ya ekle.
```

---

### GÖREV 5: RetrievalEvaluator

**PROMPT:**
```
AGENTS.md'yi oku.

GÖREV: src/evaluators/ içine retrieval metrikleri ve evaluator yaz.

1. src/evaluators/metrics.py → PURE fonksiyonlar (class yok, sadece fonksiyonlar)
   - precision_at_k(retrieved_ids, relevant_ids, k) → float
   - recall_at_k(retrieved_ids, relevant_ids, k) → float
   - mrr(retrieved_ids, relevant_ids) → float
   - ndcg_at_k(retrieved_ids, relevant_ids, k) → float
   - Tüm fonksiyonlar type hint'li ve docstring'li
   - Edge case: boş liste → 0.0

2. src/evaluators/config.py → RetrievalEvaluatorConfig
   - k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

3. src/evaluators/retrieval_evaluator.py → RetrievalEvaluator
   - evaluate(retrieval_result, ground_truth_ids) → Dict[str, float]
   - Dönen dict: {"precision@1": 0.8, "precision@3": 0.6, "mrr": 1.0, ...}
   - Tüm k_values için tüm metrikleri hesapla

4. src/evaluators/__init__.py → export'lar

5. tests/evaluators/test_metrics.py → her metrik için birim testler
   - Elle hesaplanmış beklenen değerlerle karşılaştır
   - test_precision_at_k_with_partial_match
   - test_recall_at_k_with_missing_relevant
   - test_mrr_when_first_result_is_correct
   - test_mrr_when_correct_result_is_third
   - test_ndcg_at_k_perfect_ranking
   - test_all_metrics_return_zero_for_empty_retrieved

Formüller:
- Precision@K = |retrieved[:K] ∩ relevant| / K
- Recall@K = |retrieved[:K] ∩ relevant| / |relevant|
- MRR = 1 / rank_of_first_relevant  (bulunamazsa 0)
- nDCG@K = DCG@K / IDCG@K  (DCG = Σ rel_i / log2(i+1))
```

---

### GÖREV 6: RAGPipeline

**PROMPT:**
```
AGENTS.md'yi oku.
Şu ana kadar tamamlanan tüm modülleri oku.

GÖREV: src/pipeline/ klasörüne RAGPipeline yaz.

1. src/pipeline/config.py → RAGPipelineConfig
   - top_k: int = 5
   - evaluate_retrieval: bool = True
   - evaluate_generation: bool = False

2. src/pipeline/result.py → PipelineResult dataclass
   - @dataclass(frozen=True)
   - query: Query
   - rag_response: RAGResponse
   - retrieval_metrics: Optional[Dict[str, float]]
   - total_latency_seconds: float

3. src/pipeline/rag_pipeline.py → RAGPipeline
   - __init__(retriever, generator, config, retrieval_evaluator=None)
   - run(query, ground_truth_ids=None) → PipelineResult
     Sıra: retrieve → generate → (evaluate) → PipelineResult
   - run_batch(queries, ground_truths=None) → List[PipelineResult]

4. src/pipeline/__init__.py → export'lar
5. tests/pipeline/test_rag_pipeline.py → testler (retriever ve generator mock'la)
```

---

### GÖREV 7: config_loader.py

**PROMPT:**
```
AGENTS.md'yi oku.

GÖREV: src/utils/config_loader.py yaz.

YAML config dosyalarını okuyup ilgili Python config nesnelerine çeviren yardımcı.

1. src/utils/config_loader.py
   - load_experiment_config(path: str) → dict
     YAML dosyasını oku, dict olarak dön
   - build_retriever_config(config_dict: dict) → PineconeRetrieverConfig | ChromaRetrieverConfig
     "type" alanına göre doğru config sınıfını oluştur
   - build_embedder_config(config_dict: dict) → SentenceTransformersEmbedderConfig
   - build_generator_config(config_dict: dict) → OpenAIGeneratorConfig
   - build_chunker_config(config_dict: dict) → FixedSizeChunkerConfig
   - build_pipeline_config(config_dict: dict) → RAGPipelineConfig
   - PyYAML kullan

2. src/utils/__init__.py → export'lar
3. tests/utils/test_config_loader.py → testler
   - experiments/configs/baseline_pinecone_openai.yaml'ı yükleyip doğrula
```

---

### GÖREV 8: run_experiment.py

**PROMPT:**
```
AGENTS.md'yi oku.
src/utils/config_loader.py ve src/pipeline/rag_pipeline.py'yi oku.

GÖREV: scripts/run_experiment.py CLI script'i yaz.

Kullanım:
  python scripts/run_experiment.py --config experiments/configs/baseline_pinecone_openai.yaml
  python scripts/run_experiment.py --config experiments/configs/baseline_chroma_openai.yaml --dry-run

Yaptıkları:
1. --config YAML'ını oku
2. config_loader ile tüm component'leri oluştur
3. Dataset'i yükle, dokümanları chunk'la, embed et, retriever'a yükle
4. Pipeline'ı çalıştır (tüm queriler için)
5. Ortalama metrikleri hesapla
6. experiments/results/{timestamp}_{experiment_name}.json olarak kaydet
7. Terminale özet tablo yazdır

Output JSON:
{
  "experiment_name": "...",
  "timestamp": "...",
  "config_path": "...",
  "metrics": {
    "precision@1": 0.72,
    "recall@5": 0.85,
    "mrr": 0.78,
    "avg_retrieval_latency_seconds": 0.23
  },
  "num_queries": 100,
  "per_query_results": [...]
}

argparse kullan. --config zorunlu, --dry-run opsiyonel (config yükler ama çalıştırmaz).
```

---

### GÖREV 9: ChromaRetriever

**PROMPT:**
```
AGENTS.md'yi oku.
src/retrievers/pinecone_retriever.py'yi oku (birebir aynı interface'i kullanacaksın).

GÖREV: src/retrievers/chroma_retriever.py yaz.

Pinecone ile AYNI Retriever interface'ini implemente et.
chromadb paketi kullan (pip install chromadb).
Fark: local çalışır, API key gerekmez.

1. src/retrievers/config.py'ye ekle → ChromaRetrieverConfig
   - collection_name: str
   - persist_directory: Optional[str] = None  (None = in-memory)
   - distance_metric: str = "cosine"
   - dimension: int

2. src/retrievers/chroma_retriever.py → ChromaRetriever
   - add_chunks(chunks, embeddings)
   - retrieve(query, top_k)
   - retrieve_with_embedding(query_embedding, top_k, query_id)
   - clear()
   - Lazy initialization: client ilk kullanımda oluşsun

3. src/retrievers/registry.py'ye ChromaRetriever'ı register et

4. tests/retrievers/test_chroma_retriever.py → testler (chromadb mock'la)

AMAÇ: run_experiment.py'de sadece config değişince
Pinecone yerine Chroma çalışsın — pipeline kodu değişmesin.
```

---

### GÖREV 10: GenerationEvaluator

**PROMPT:**
```
AGENTS.md'yi oku.

GÖREV: src/evaluators/generation_evaluator.py yaz.
Bu evaluator, bir LLM'i "hakem" olarak kullanır (LLM-as-a-judge).

1. src/evaluators/judge_prompts.py → Sabit prompt şablonları
   FAITHFULNESS_JUDGE_PROMPT: str
     → Verilen context ve cevaba bakarak, cevabın context'e sadakatini 0-10 arasında puan ver.
        JSON formatında dön: {"score": 8, "reason": "..."}
   RELEVANCY_JUDGE_PROMPT: str
     → Soruya cevabın ne kadar alakalı olduğunu 0-10 puan ver.
        JSON: {"score": 7, "reason": "..."}

2. src/evaluators/generation_evaluator.py → GenerationEvaluator
   - __init__(judge_generator: Generator, config)
   - evaluate(rag_response: RAGResponse) → Dict[str, float]
     Dönen: {"faithfulness": 0.8, "answer_relevancy": 0.7}
   - Hakem için Generator interface'ini kullan (hardcode openai değil)
   - LLM'den gelen JSON'u parse et, 0-1 aralığına normalize et

3. tests/evaluators/test_generation_evaluator.py → testler
   - judge generator'ı mock'la
   - test_evaluate_returns_scores_between_0_and_1
   - test_faithfulness_prompt_contains_context_and_answer
```

---

## 🏛️ MİMARİ KURALLAR

### Proje Nedir?
RAG sistemlerini farklı vektör DB'ler ve LLM'lerle karşılaştırmalı değerlendiren yüksek lisans tezi framework'ü.

### Klasör → Sorumluluk

| Klasör | Ne Yapar | Ne Yapamaz |
|--------|----------|-----------|
| `src/core/` | Abstract interface'ler ve tipler | Hiçbir implementasyon |
| `src/chunkers/` | Chunker implementasyonları | Generator, retriever kodu |
| `src/embedders/` | Embedder implementasyonları | Retrieval, generation kodu |
| `src/retrievers/` | Retriever implementasyonları | Generator, evaluator kodu |
| `src/generators/` | Generator implementasyonları | Retriever, evaluator kodu |
| `src/evaluators/` | Metrik hesaplama | Pipeline, component kodu |
| `src/datasets/` | Dataset loader'lar | Component implementasyonları |
| `src/pipeline/` | Her şeyi compose eder | Component implementasyonları |

### Kırılmaz 5 Kural

1. **Hardcode yasak** — model adı, prompt, API key asla koda yazılmaz
2. **Frozen dataclass** — tüm config ve type sınıfları `@dataclass(frozen=True)`
3. **Çapraz import yasak** — `retrievers/` sadece `core/` ve `utils/`'ten import eder
4. **Type hint zorunlu** — her parametre ve return type belirtilir
5. **Evaluation leak yasak** — metrik kodu pipeline'ın içine girmez

### Referans Dosyalar
- Implementasyon stili → `src/retrievers/pinecone_retriever.py`
- Config pattern → `src/retrievers/config.py`
- Registry pattern → `src/retrievers/registry.py`
- Test pattern → `tests/retrievers/test_pinecone_retriever.py`

### Environment Variables
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

---

## 📝 GÜNCELLEME TALİMATI

Bir görev bitince şunu yap:

1. `YAPILACAKLAR` listesinde `- [ ]` → `- [x]` yap
2. `TAMAMLANAN MODÜLLER` bölümüne ekle
3. `AKTİF GÖREV` bölümünü bir sonrakiyle değiştir
4. `SONRAKI GÖREV` bölümünü bir sonraki için güncelle
5. `SON GÜNCELLEME` tarihini yaz