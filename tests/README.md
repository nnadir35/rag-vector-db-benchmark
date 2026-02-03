# Test Suite

Bu dizin RAG benchmark framework'ün test dosyalarını içerir.

## Test Yapısı

```
tests/
├── conftest.py                    # Shared fixtures ve mock'lar
├── retrievers/
│   ├── test_pinecone_retriever.py # PineconeRetriever testleri
│   ├── test_registry.py           # Registry pattern testleri
│   └── test_config.py             # Config class testleri
```

## Test Çalıştırma

### Tüm Testleri Çalıştırma

```bash
# Proje root dizininden
pytest tests/

# Veya verbose modda
pytest tests/ -v
```

### Belirli Bir Test Dosyasını Çalıştırma

```bash
# Sadece PineconeRetriever testleri
pytest tests/retrievers/test_pinecone_retriever.py

# Sadece registry testleri
pytest tests/retrievers/test_registry.py

# Sadece config testleri
pytest tests/retrievers/test_config.py
```

### Belirli Bir Test Class'ını Çalıştırma

```bash
# Sadece initialization testleri
pytest tests/retrievers/test_pinecone_retriever.py::TestPineconeRetrieverInitialization
```

### Belirli Bir Test Fonksiyonunu Çalıştırma

```bash
# Sadece bir test
pytest tests/retrievers/test_pinecone_retriever.py::TestPineconeRetrieverInitialization::test_init_success
```

### Coverage Raporu

```bash
# Coverage ile çalıştırma
pytest tests/ --cov=src --cov-report=html

# Coverage raporunu görüntüleme
open htmlcov/index.html  # macOS
# veya
xdg-open htmlcov/index.html  # Linux
```

## Test Stratejisi

### Mock Kullanımı

Testler external dependency'leri (Pinecone API) mock'layarak çalışır:

- **MockEmbedder**: Deterministic embedding'ler üretir
- **MockPineconeIndex**: Pinecone index'i simüle eder
- **MockPineconeClient**: Pinecone client'ı simüle eder

Bu sayede:

- Testler hızlı çalışır (API çağrısı yok)
- Deterministic'tir (her çalıştırmada aynı sonuç)
- CI/CD'de çalışabilir (internet bağlantısı gerekmez)
- Ücretsizdir (API maliyeti yok)

### Test Kategorileri

1. **Initialization Tests**: Retriever'ın doğru şekilde initialize edilmesi
2. **Add Chunks Tests**: Chunk ekleme işlevselliği
3. **Retrieve Tests**: Retrieval işlevselliği
4. **Clear Tests**: Index temizleme işlevselliği
5. **Metadata Tests**: Metadata'nın korunması
6. **Error Handling Tests**: Hata durumlarının doğru handle edilmesi

## Gereksinimler

Testleri çalıştırmak için:

```bash
pip install pytest pytest-cov
```

Opsiyonel (coverage için):

```bash
pip install pytest-cov
```

## Yeni Test Ekleme

Yeni bir component için test eklerken:

1. `tests/<component_name>/` dizini oluştur
2. `test_<component_name>.py` dosyası oluştur
3. `conftest.py`'ye gerekli fixture'ları ekle
4. Test class'larını ve fonksiyonlarını yaz
5. Mock'ları kullan (external dependency'ler için)

Örnek:

```python
class TestMyComponent:
    def test_something(self, fixture_name):
        # Test code
        assert something == expected
```
