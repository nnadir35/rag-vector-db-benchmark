# Docker ile çalıştırma

Bu düzen **Qdrant**’ı ayrı bir konteynerde (gerçek servis), **FastAPI**’yi `rag-app` içinde çalıştırır. Chroma verisi isteğe bağlı olarak `/data/chroma` volume’ünde kalır (`CHROMA_PERSIST_DIRECTORY`).

## Önkoşullar

- Docker Engine 20.10+ ve Docker Compose V2
- (İsteğe bağlı) Host’ta Ollama — `experiments/configs/docker_rag_api.yaml` içindeki `generator.api_base` varsayılan olarak `http://host.docker.internal:11434` kullanır; Linux’ta `docker-compose.yml` içindeki `extra_hosts` satırı bunu sağlar.

## Ayağa kaldırma

Proje kökünde:

```bash
docker compose up --build
```

Arka planda:

```bash
docker compose up --build -d
```

- **Qdrant HTTP:** `http://localhost:6333` (konteynerler arası: `http://qdrant-server:6333` veya `http://qdrant_db:6333`)
- **FastAPI:** `http://localhost:8000` — örn. `http://localhost:8000/docs`
- **ChromaDB:** `http://localhost:8000` (FastAPI ile çakışabilir, portu `CHROMA_PORT` ile ayarlayın)
- **ElasticSearch:** `http://localhost:9200`
- **Milvus:** `http://localhost:19530` (API port: 9091)
- **Postgres:** `localhost:5432`
- **Redis:** `localhost:6379`

## Ortam değişkenleri (özet)

| Değişken | Açıklama |
|----------|----------|
| `QDRANT_URL` | Tam URL (öncelikli), örn. `http://qdrant-server:6333` |
| `QDRANT_HOST` | Sunucu host’u (Compose’ta `qdrant_db`) |
| `QDRANT_PORT` | Varsayılan `6333` |
| `CHROMA_PERSIST_DIRECTORY` | Chroma disk yolu (örn. `/data/chroma`) |
| `CHROMA_HOST` / `CHROMA_PORT` | Ayrı Chroma sunucusu (varsayılan port `8000`) |
| `RAG_CONFIG_PATH` | YAML yolu (Compose’ta `experiments/configs/docker_rag_api.yaml`) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Postgres veritabanı ayarları |
| `REDIS_PORT` | Redis portu (varsayılan `6379`) |
| `ELASTIC_VERSION` / `ELASTIC_PORT` | ElasticSearch versiyon ve portu (varsayılan `8.12.0`, `9200`) |
| `MILVUS_PORT` / `MILVUS_API_PORT` | Milvus portları (varsayılan `19530`, `9091`) |

## Servis Bağımlılıkları (Startup Sequence)

`docker-compose.yml` içinde bazı servisler diğerlerinin ayağa kalkmasını bekler:
- **Milvus (`milvus_db`)**: Çalışmadan önce `etcd` (`milvus_etcd`) ve `minio` (`milvus_minio`) servislerinin `healthy` durumuna gelmesini bekler (`depends_on` ve `healthcheck` mekanizması ile).
- Diğer veritabanları (Qdrant, Chroma, ElasticSearch, Postgres, Redis) birbirinden bağımsız kalkar ve kendi `healthcheck` tanımlarına sahiptir.

## Benchmark’ı konteyner içinden çalıştırma

`rag-app` ortamında `QDRANT_HOST=qdrant-server` tanımlı olduğu için `scripts/benchmark_db.py` içindeki Qdrant istemcisi **Compose ağındaki Qdrant servisine** gider (Docker üzerinden gecikme / izolasyon ölçümü için uygun):

```bash
docker compose exec rag-app python scripts/benchmark_db.py --num-documents 500 --num-queries 50 --no-progress
```

Yerel gömülü (`:memory:`) Qdrant ile karşılaştırmak için aynı komutta `QDRANT_HOST` ve `QDRANT_URL` boşaltılabilir:

```bash
docker compose exec -e QDRANT_HOST= -e QDRANT_URL= rag-app python scripts/benchmark_db.py --num-documents 200 --num-queries 30 --no-progress
```

## Durdurma

```bash
docker compose down
```

Verileri de silmek için: `docker compose down -v` (Qdrant ve Chroma volume’leri gider).
