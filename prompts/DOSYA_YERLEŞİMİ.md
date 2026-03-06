# DOSYA_YERLEŞİMİ.md — Her Dosya Nereye Gidecek

## Projenin Tam Klasör Yapısı (Hedef)

```
rag-vector-db-benchmark/
│
├── AGENTS.md                          ← ✅ YENİ — AI asistan için ana talimat dosyası
├── PROMPTS.md                         ← ✅ YENİ — Hazır prompt şablonları
├── TEORİ.md                           ← ✅ YENİ — Öğrenmen gereken minimum teori
├── DOSYA_YERLEŞİMİ.md                 ← ✅ YENİ — Bu dosya
├── CONTEXT.md                         ← Mevcut (değiştirme)
├── STRUCTURE.md                       ← Mevcut (değiştirme)
├── README.md                          ← Mevcut (değiştirme)
├── .cursorrules                       ← ✅ YENİ — Cursor/Windsurf için güncellenmiş kurallar
│
├── src/
│   ├── __init__.py                    ← Mevcut
│   │
│   ├── core/                          ← Mevcut, dokunma
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── generation.py
│   │   ├── ingestion.py
│   │   └── retrieval.py
│   │
│   ├── chunkers/                      ← ✅ YENİ KLASÖR — Prompt #1 ile yap
│   │   ├── __init__.py
│   │   ├── config.py                  ← FixedSizeChunkerConfig
│   │   └── fixed_size_chunker.py      ← FixedSizeChunker
│   │
│   ├── embedders/                     ← ✅ YENİ KLASÖR — Prompt #2 ile yap
│   │   ├── __init__.py
│   │   ├── config.py                  ← SentenceTransformersEmbedderConfig
│   │   └── sentence_transformers_embedder.py
│   │
│   ├── retrievers/                    ← Mevcut
│   │   ├── __init__.py
│   │   ├── config.py                  ← Mevcut: PineconeRetrieverConfig
│   │   │                                ✅ Ekle: ChromaRetrieverConfig (Prompt #9)
│   │   ├── pinecone_retriever.py      ← Mevcut
│   │   ├── chroma_retriever.py        ← ✅ YENİ — Prompt #9 ile yap
│   │   └── registry.py               ← Mevcut (chroma'yı register et)
│   │
│   ├── generators/                    ← ✅ Doldurulacak — Prompt #3 ile yap
│   │   ├── __init__.py
│   │   ├── config.py                  ← OpenAIGeneratorConfig
│   │   ├── openai_generator.py        ← OpenAIGenerator
│   │   └── registry.py               ← registry.py'yi kopyala, düzenle
│   │
│   ├── evaluators/                    ← ✅ Doldurulacak — Prompt #4 ve #5 ile yap
│   │   ├── __init__.py
│   │   ├── config.py                  ← RetrievalEvaluatorConfig
│   │   ├── metrics.py                 ← Pure fonksiyonlar: precision_at_k, mrr, vb.
│   │   ├── retrieval_evaluator.py     ← RetrievalEvaluator
│   │   ├── generation_evaluator.py    ← GenerationEvaluator (LLM-as-judge)
│   │   └── judge_prompts.py           ← Hakem prompt şablonları
│   │
│   ├── datasets/                      ← ✅ Doldurulacak — Prompt #7 ile yap
│   │   ├── __init__.py
│   │   ├── config.py                  ← SQuADDatasetConfig
│   │   └── squad_loader.py            ← SQuADLoader
│   │
│   ├── pipeline/                      ← ✅ Doldurulacak — Prompt #6 ile yap
│   │   ├── __init__.py
│   │   ├── config.py                  ← RAGPipelineConfig
│   │   └── rag_pipeline.py            ← RAGPipeline
│   │
│   └── utils/                         ← Sonra ekle, acil değil
│       ├── __init__.py
│       ├── logging.py                 ← Logging setup
│       └── config_loader.py           ← YAML → config nesnesi
│
├── experiments/
│   ├── configs/
│   │   ├── baseline_pinecone_openai.yaml    ← ✅ YENİ — hazır
│   │   ├── baseline_chroma_openai.yaml      ← ✅ YENİ — hazır
│   │   └── ablation_chunk_size_256.yaml     ← ✅ YENİ — hazır
│   │
│   └── results/                       ← run_experiment.py buraya yazar
│       └── .gitkeep
│
├── tests/
│   ├── __init__.py                    ← Mevcut
│   ├── conftest.py                    ← Mevcut (MockEmbedder burada)
│   │
│   ├── chunkers/                      ← ✅ Prompt #1 ile oluşturulacak
│   │   ├── __init__.py
│   │   └── test_fixed_size_chunker.py
│   │
│   ├── embedders/                     ← ✅ Prompt #2 ile oluşturulacak
│   │   ├── __init__.py
│   │   └── test_sentence_transformers_embedder.py
│   │
│   ├── retrievers/                    ← Mevcut
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_pinecone_retriever.py
│   │   └── test_registry.py
│   │
│   ├── generators/                    ← ✅ Prompt #3 ile oluşturulacak
│   │   ├── __init__.py
│   │   └── test_openai_generator.py
│   │
│   ├── evaluators/                    ← ✅ Prompt #4 ile oluşturulacak
│   │   ├── __init__.py
│   │   ├── test_metrics.py
│   │   └── test_retrieval_evaluator.py
│   │
│   └── pipeline/                      ← ✅ Prompt #6 ile oluşturulacak
│       ├── __init__.py
│       └── test_rag_pipeline.py
│
├── scripts/
│   └── run_experiment.py              ← ✅ Prompt #10 ile oluşturulacak
│
└── docs/
    ├── LEARNING_PATH.md               ← ✅ YENİ — haftalık öğrenme planı
    └── architecture.md                ← Sonra ekle
```

---

## Öncelik Sırası (Hangi Sıraya Uy)

Her adım bir öncekine bağlı. Atlamadan git.

```
[1] src/chunkers/              → Prompt #1
     ↓
[2] src/embedders/             → Prompt #2
     ↓
[3] src/generators/            → Prompt #3
     ↓
[4] src/evaluators/metrics.py  → Prompt #4 (önce pure functions)
     ↓
[5] src/evaluators/retrieval_evaluator.py → Prompt #4 (evaluator class)
     ↓
[6] src/pipeline/              → Prompt #6
     ↓
[7] src/datasets/              → Prompt #7
     ↓
[8] scripts/run_experiment.py  → Prompt #10
     ↓
[9] src/retrievers/chroma_retriever.py → Prompt #9
     ↓
[10] Karşılaştırmalı deneyleri çalıştır
```

---

## Hangi Dosyaları AI'ya Göstereceksin?

Her seferinde bu dosyaları context olarak ver:
1. `AGENTS.md` (zorunlu — her seferinde)
2. `src/core/[ilgili_interface].py` (implement edilecek interface)
3. `src/retrievers/pinecone_retriever.py` (implementasyon referansı)
4. `src/retrievers/config.py` (config pattern referansı)
