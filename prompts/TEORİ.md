# TEORİ.md — RAG için Bilmen Gereken Minimum Teori

## ML Öğrenmene Gerek Var mı?

**Hayır.** Bu proje için gereken bilgi çok spesifik.

Şöyle düşün: Bir web developer React component'lerinin DOM'da nasıl render edildiğinin
C++ implementasyonunu bilmek zorunda değil. Sen de embedding modellerinin
transformer mimarisini bilmek zorunda değilsin.

**Bilmen gerekenler:** Embedding ne işe yarar, neden kullanılıyor, nasıl karşılaştırılır.
**Bilmene gerek yok:** Transformer, backpropagation, gradient descent, sinir ağı matematiği.

---

## Bölüm 1: Embedding — "Anlam → Sayı"

### Ne Oluyor?

Bir embedding modeli metni alır, bir sayı dizisine (vektöre) çevirir.

```
"Köpek" → [0.2, -0.8, 0.5, 0.1, ...]   (384 sayı)
"Kedi"  → [0.3, -0.7, 0.4, 0.2, ...]   (384 sayı — köpeğe yakın!)
"Araba" → [0.9,  0.2, -0.3, 0.8, ...]  (384 sayı — köpekten uzak)
```

Anlamca benzer kelimeler/cümleler, uzayda birbirine yakın noktalar oluyor.

### Neden Bu Projeyle İlgili?

Pinecone'a `add_chunks()` dediğinde ne yapıyorsun aslında?
1. Her chunk'ı embedding modeline gönderiyorsun
2. 384 boyutlu vektör alıyorsun
3. Bu vektörü Pinecone'a kaydediyorsun

`retrieve(query)` dediğinde ne oluyor?
1. Query de embedding modeline giriyor → vektöre dönüşüyor
2. Pinecone, query vektörüne en yakın chunk vektörlerini buluyor
3. "Yakınlık" = anlamsal benzerlik

### Cosine Similarity — Neden?

İki vektörün ne kadar aynı yönde baktığını ölçer.
- 1.0 = tamamen aynı anlam
- 0.0 = alakasız
- -1.0 = zıt anlam

```python
# PineconeRetrieverConfig'de metric="cosine" diyorsun
# Bu yüzden — cosine similarity semantik arama için en yaygın
config = PineconeRetrieverConfig(
    metric="cosine",  # ← bu
    dimension=384,    # ← embedding modelinin boyutu
)
```

### Tez Savunmasında Sana Sorulabilecek

**S: Neden all-MiniLM-L6-v2 modeli seçtiniz?**
C: Hız/kalite dengesi açısından MTEB benchmark'larında güçlü sonuçlar veriyor,
384 boyutlu küçük vektörlerle düşük depolama maliyeti sağlıyor. Daha büyük modeller
(text-embedding-3-large gibi) daha iyi recall verebilir ama maliyeti yüksek.

---

## Bölüm 2: Chunking — "Neden Küçük Parçalara Bölüyoruz?"

### Problem

Elimizdeki doküman 50 sayfa. Embedding modeli max 512 token kabul ediyor.
50 sayfalık metni tek vektöre dökürsek anlam gider. Çok genel bir vektör olur.

### Çözüm

Belgeyi küçük parçalara böl (chunk), her parçayı ayrı ayrı embed et.

### Chunk Size Tradeoff'u

| Küçük chunk (128 token) | Büyük chunk (512 token) |
|------------------------|------------------------|
| Spesifik, odaklı | Daha fazla bağlam içeriyor |
| Retrieval precison ↑ | Retrieval recall ↑ |
| Cevap için bağlam az | Noise fazla |

**Overlap neden var?**
```
Chunk 1: "Ali markete gitti. Elma aldı."
Chunk 2: "Elma aldı. Eve döndü."  ← "Elma aldı" tekrar ediyor
```
Cümle sınırında kesilen bir bilgi kaybolmasın diye.

### Tez Bağlantısı

Projenin adı "benchmark" — chunk_size'ın retrieval kalitesine etkisini ölçmek
tezinin bir katkısı olabilir. Küçük vs büyük chunk deneyleri yapabilirsin.

---

## Bölüm 3: Retrieval Metrikleri — Tezinin Kalbi

Bu metrikleri tam anlamalısın. Tezinde "hangi vektör DB daha iyi" sorusunun
cevabı bunlarla verilecek.

### Setup

```
Sorumuz: "İklim değişikliği nedir?"

Gerçekten alakalı dokümanlar (ground truth): {doc_A, doc_B, doc_C}

Pinecone'un getirdikleri (top 5): [doc_A, doc_X, doc_B, doc_Y, doc_Z]
                                     1       2       3       4       5
```

### Precision@K — "Getirdiklerinin kaçı doğru?"

```
Precision@3 = Alakalı olan (ilk 3 içinde) / 3
            = 2 (doc_A ve doc_B) / 3
            = 0.67

Precision@5 = 2 / 5 = 0.40
```

Yüksek Precision@K → Gereksiz sonuç az, kalite yüksek.

### Recall@K — "Doğruların kaçını buldun?"

```
Recall@5 = Alakalı olan (ilk 5 içinde) / Toplam alakalı
         = 2 (doc_A ve doc_B bulundu, doc_C bulunamadı) / 3
         = 0.67
```

Yüksek Recall@K → Sistemin az şeyi kaçırması.

### MRR — "İlk doğru sonuç kaçıncı sırada?"

```
MRR = 1 / (İlk doğru sonucun sırası)
    = 1 / 1   (doc_A zaten 1. sırada!)
    = 1.0
```

Sistemin en üste doğru şeyi koyması → MRR yüksek.

### nDCG@K — "Sıralamanın kalitesi"

Üstteki sonuçlar daha değerli kabul edilir. doc_A 1. sırada gelirse 5. sıradakinden daha fazla puan alır.

### Pratikte Ne Anlıyoruz?

| Senaryo | Önemli Metrik |
|---------|--------------|
| Soru-cevap sistemi | MRR (ilk sonuç doğru mu?) |
| Genel doküman arama | Recall@10 (kaçırmama önemli) |
| LLM'e context verme | Precision@5 (noise az olsun) |
| Benchmark karşılaştırma | nDCG@10 (bütüncül sıralama kalitesi) |

---

## Bölüm 4: RAG Generation — "Retrieval Sonrası Ne Oluyor?"

### Flow

```python
query = "İklim değişikliği neden oluyor?"

# 1. Retrieval — sen zaten yazdın
retrieved_chunks = retriever.retrieve(query, top_k=5)

# 2. Prompt assembly — generator yapıyor
prompt = f"""
Context:
[1] {retrieved_chunks[0].chunk.content}
[2] {retrieved_chunks[1].chunk.content}
...

Soru: {query.text}
Sadece yukarıdaki bağlamı kullanarak cevapla.
"""

# 3. LLM çağrısı — generator yapıyor
response = openai.chat.completions.create(model="gpt-4o-mini", messages=[...])
```

### Neden temperature=0.0?

Araştırma projesinde **reproducibility** önemli.
temperature=0 → her çalıştırmada aynı cevap → sonuçlar karşılaştırılabilir.

### Hallucination Nedir?

LLM context'te olmayan bir şeyi uydurursa → hallucination.
"Faithfulness" metriği bunu ölçer: cevap gerçekten context'ten mi geliyor?

---

## Bölüm 5: Tez İçin Okuman Gereken 3 Şey

Hepsini okumana gerek yok. Sadece şunlar:

### 1. Lewis et al. 2020 — Orijinal RAG Makalesi
**Ne kadarını okuyacaksın:** Abstract + Section 1 (Introduction) + Section 4 (Experiments başlığı)
**Neden:** "RAG nedir" sorusunu tezinde buraya referans vererek cevaplayacaksın.
**Link:** arxiv.org/abs/2005.11401

### 2. RAGAS Paper — Generation Evaluation
**Ne kadarını okuyacaksın:** Abstract + Table 1 (metrikler)
**Neden:** generation_evaluator.py yazarken hangi metrikleri ölçeceğini anlarsın.
**Link:** arxiv.org/abs/2309.15217

### 3. BEIR Benchmark Paper
**Ne kadarını okuyacaksın:** Abstract + Section 2 (Tasks)
**Neden:** Projenin benchmark felsefesi bununla örtüşüyor. Tezde "BEIR yaklaşımına benzer şekilde" diyebilirsin.
**Link:** arxiv.org/abs/2104.08663

---

## Tez Savunmasında Muhtemelen Sorulacaklar

**S: Neden Pinecone ve Chroma'yı karşılaştırdınız?**
C: Managed cloud (Pinecone) vs local/open-source (Chroma) tradeoff'unu incelemek istedim.
Production maliyeti, latency ve retrieval kalitesi açısından farkları ölçtüm.

**S: Chunk size seçiminizi nasıl yaptınız?**
C: 256, 512, 1024 token için deneyleri çalıştırdım, [sonuç metrikleri] ışığında 512'nin
bu dataset için en iyi precision/recall dengesini verdiğini gördüm.

**S: Evaluation nasıl ground truth'a ihtiyaç duyuyor?**
C: SQuAD dataset'inde her soru bir context paragrafına ait. Ground truth = o paragrafın ID'si.
Ideal retriever, soruyu query olarak verdiğimde o paragrafı top-1'e getirmeli.
Precision@1, MRR bunu ölçüyor.

**S: Retrieval ve generation'ı neden ayrı değerlendirdiniz?**
C: Kötü bir cevabın kaynağı belirsiz kalıyor: retriever yanlış doküman mı getirdi,
yoksa generator doğru dokümanı yanlış mı yorumladı? Ayrı değerlendirme bottleneck'i tespit etmemi sağladı.
