# docs/LEARNING_PATH.md — Ne Zaman Ne Öğreneceksin

## ML Öğrenmene Gerek Yok

RAG sistemi bir **arama + özetleme** pipeline'ı.
Embedding modeli senin yazmadığın bir araç — sadece nasıl kullanıldığını anlaman yeterli.

---

## Haftalık Öğrenme Planı (Kod + Teori Paralel)

### Hafta 1 — Chunking + Embedding

**Kod:** `FixedSizeChunker` + `SentenceTransformersEmbedder` yaz

**Teori (30 dk):**
- sentence-transformers.net → "Pretrained Models" sayfası
  → "all-MiniLM-L6-v2" ne kadar boyut, ne kadar hızlı? Anla.
- "What is a vector embedding?" — Pinecone blog, ~10 dk okuma

**Soru sor kendine:** `dimension=384` nereden geliyor? Değiştirsem ne olur?

---

### Hafta 2 — Generator

**Kod:** `OpenAIGenerator` yaz

**Teori (45 dk):**
- OpenAI API docs → Chat Completions kısmı → token nedir, cost nasıl hesaplanır?
- "RAG vs Fine-tuning" — Pinecone blog veya LangChain docs
  → Ne zaman RAG, ne zaman fine-tuning? Tezde bu soruyu cevapla.

**Anla:** `temperature=0.0` neden? `max_tokens` neden sınırlı?

---

### Hafta 3 — Retrieval Evaluator

**Kod:** `RetrievalEvaluator` + `metrics.py` yaz

**Teori (1 saat — bu hafta biraz fazla ama şart):**
- Wikipedia: "Information Retrieval Evaluation" → Precision, Recall, F1 bölümleri
- "NDCG explained" — Medium'da iyi bir makale var, 15 dk
- Lewis et al. 2020 RAG makalesi → Abstract + Section 1 oku

**Elle hesapla:** 5 sonuçtan 2'si doğruysa Precision@5, Recall@5, MRR kaç?
Sonra kende hesaplamayı kodla karşılaştır.

---

### Hafta 4 — Pipeline + İlk Deney

**Kod:** `RAGPipeline` + `run_experiment.py` yaz

**Teori (30 dk):**
- BEIR paper → Abstract + Table 1 (hangi datasetler, hangi metrikler)
- SQuAD dataset nedir? → HuggingFace dataset kartı oku

**İlk gerçek deney:** 10 soruyla `baseline_chroma_openai.yaml` çalıştır.
Metrikleri gör. Anlamlı mı? Düşük mü? Neden?

---

### Hafta 5 — Karşılaştırma + Analiz

**Kod:** `ChromaRetriever` yaz, Pinecone ile karşılaştır

**Teori (45 dk):**
- RAGAS paper → Abstract + Section 3 (metrikler)
- "Statistical significance in ML experiments" — tez için önemli,
  iki sistemin farkı anlamlı mı? t-test veya basit confidence interval.

---

### Hafta 6–8 — Tez Yazımı

Artık yeni teori yok. Deney sonuçlarını analiz et ve yaz.

---

## "Bunu Bilmek Zorunda Mıyım?" Rehberi

| Konu | Gerekli mi? | Neden |
|------|-------------|-------|
| Transformer mimarisi (attention, BERT) | ❌ Hayır | Modeli sadece kullanıyorsun |
| Backpropagation, gradient descent | ❌ Hayır | Model eğitmiyorsun |
| Word2Vec, GloVe tarihi | ❌ Hayır | Modern embedding farklı |
| Cosine similarity hesabı | ✅ Evet | Neden kullandığını açıklaman lazım |
| Precision@K, Recall@K formülü | ✅ Evet | Tezinin evaluation kısmı |
| Tokenization nedir | ✅ Evet | max_tokens açıklarken lazım |
| Prompt engineering temelleri | ✅ Evet | Generator config için |
| RAG vs Fine-tuning farkı | ✅ Evet | Tez motivasyonu için |
| Vector DB indexing (HNSW, IVF) | 🟡 Opsiyonel | Derinlemesine benchmark için |
| Chunk size etkisi | ✅ Evet | Ablation study yapıyorsan |
