# AGENTS.md — AI Asistan Talimatları

Bu dosyayı her yeni sohbetin başında Claude veya Cursor'a ver.
Projenin tamamını anlatır. Okumadan kod yazma.

---

## Proje Nedir?

RAG (Retrieval-Augmented Generation) sistemlerini farklı vektör veritabanları,
embedding modelleri ve LLM'ler ile **karşılaştırmalı olarak değerlendiren** bir
yüksek lisans tez framework'ü.

Amacımız: "Hangi vektör DB + embedding modeli kombinasyonu en iyi retrieval
kalitesini veriyor?" sorusunu tekrarlanabilir deneylerle yanıtlamak.

---

## Mimari Özet

```
Document → Chunker → Embedder → VectorDB (Retriever)
                                      ↓
Query ──────────────────────────→ Retriever → Generator → Response
                                      ↓              ↓
                              RetrievalEvaluator  GenerationEvaluator
                                      ↓              ↓
                                    Metrics       Metrics
```

### Klasör → Sorumluluk

| Klasör | Sorumluluk | Ne İçermez |
|--------|-----------|------------|
| `src/core/` | Abstract interface'ler, type'lar | Hiçbir implementasyon |
| `src/chunkers/` | Chunker implementasyonları | Generator, retriever kodu |
| `src/embedders/` | Embedder implementasyonları | Retrieval, generation kodu |
| `src/retrievers/` | Retriever implementasyonları | Generator, evaluator kodu |
| `src/generators/` | Generator implementasyonları | Retriever, evaluator kodu |
| `src/evaluators/` | Metrik hesaplama | Pipeline, component kodu |
| `src/datasets/` | Dataset loader'lar | Component implementasyonları |
| `src/pipeline/` | Bileşenleri compose eder | Component implementasyonları |
| `src/utils/` | Ortak yardımcı araçlar | Business logic |
| `experiments/configs/` | YAML deney konfigürasyonları | Kod |
| `experiments/results/` | Deney sonuçları (immutable) | Kod, config |

---

## Kritik Kurallar (İhlal Etme)

1. **Interface-first**: Her yeni component tipi önce `src/core/` içine abstract class olarak girer
2. **Sıfır hardcode**: Model adı, prompt, DB ismi asla koda yazılmaz — hepsi config'den gelir
3. **Çapraz import yasak**: `retrievers/` sadece `core/` ve `utils/`'ten import eder, `generators/`'dan etmez
4. **Evaluation leak yasak**: Metric hesaplama kodu pipeline kodunun içine girmez
5. **Frozen dataclass**: Tüm config ve type sınıfları `@dataclass(frozen=True)` olur
6. **Type hint zorunlu**: Her fonksiyonun parametresi ve return tipi belirtilir
7. **Docstring zorunlu**: Her public method "neden böyle yapıldığını" açıklar (ne yaptığını değil)

---

## Mevcut Implementasyonlar (Hazır)

- ✅ `src/core/types.py` — Tüm veri tipleri (Document, Chunk, Query, RetrievalResult, vb.)
- ✅ `src/core/chunking.py` — Chunker ABC
- ✅ `src/core/embedding.py` — Embedder ABC
- ✅ `src/core/retrieval.py` — Retriever ABC
- ✅ `src/core/generation.py` — Generator ABC
- ✅ `src/core/ingestion.py` — DocumentIngester ABC
- ✅ `src/retrievers/pinecone_retriever.py` — Pinecone implementasyonu (referans al)
- ✅ `src/retrievers/config.py` — PineconeRetrieverConfig (config pattern referansı)
- ✅ `src/retrievers/registry.py` — Registry pattern (diğer registry'lere kopyala)

---

## Eksik (Sırayla Yapılacak)

1. `src/chunkers/fixed_size_chunker.py` + `src/chunkers/config.py`
2. `src/embedders/sentence_transformers_embedder.py` + config
3. `src/generators/openai_generator.py` + config
4. `src/evaluators/retrieval_evaluator.py` — Precision@K, Recall@K, MRR, nDCG
5. `src/evaluators/generation_evaluator.py` — Faithfulness, Answer Relevancy
6. `src/pipeline/rag_pipeline.py` — End-to-end orchestration
7. `src/datasets/squad_loader.py` — SQuAD dataset loader
8. `experiments/configs/baseline_pinecone_openai.yaml`
9. `scripts/run_experiment.py`

---

## Kod Yazarken Referans Al

Yeni bir retriever yazacaksan → `src/retrievers/pinecone_retriever.py`'ye bak
Yeni bir config yazacaksan → `src/retrievers/config.py`'ye bak
Yeni bir registry yazacaksan → `src/retrievers/registry.py`'yi kopyala
Test yazacaksan → `tests/retrievers/test_pinecone_retriever.py`'ye bak
Fixture lazımsa → `tests/conftest.py`'ye bak

---

## Import Kuralları (Örnekle)

```python
# ✅ DOĞRU — chunker, sadece core'dan import eder
from src.core.chunking import Chunker
from src.core.types import Document, Chunk

# ✅ DOĞRU — pipeline, her yerden import edebilir
from src.retrievers.pinecone_retriever import PineconeRetriever
from src.generators.openai_generator import OpenAIGenerator
from src.evaluators.retrieval_evaluator import RetrievalEvaluator

# ❌ YANLIŞ — retriever, generator'ı import edemez
from src.generators.openai_generator import OpenAIGenerator  # retrievers/ içinden

# ❌ YANLIŞ — hardcoded model
model = "text-embedding-ada-002"  # config'den gelmeli
```

---

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...   # AnthropicGenerator kullanılırsa
```

---

## Test Felsefesi

- LLM ve vector DB çağrıları **her zaman mock'lanır** (gerçek API çağrısı yapma)
- `tests/conftest.py`'deki `MockEmbedder`'ı kullan
- Test isimleri: `test_[ne test ediliyor]_[hangi koşulda]_[beklenen sonuç]`
- Örnek: `test_retrieve_with_valid_query_returns_ranked_chunks`
