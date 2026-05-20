#!/usr/bin/env python3
"""Script to download and parse the SQuAD v2 dataset for objective performance measurement.

This script uses the project's internal SQuADLoader to fetch the dataset via HuggingFace,
parses it into our standardized Document and Query data structures, and ensures
deterministic ground truth mappings.
"""

import json
import logging
import sys
from pathlib import Path

# Proje kök dizinini import yollarına ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.config import SQuADDatasetConfig
from src.datasets.squad_loader import SQuADLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("SQuAD v2 dataset konfigürasyonu yükleniyor...")
    # Nesnel benchmark için 'validation' setini kullanıyoruz
    config = SQuADDatasetConfig(split="validation", version="squad_v2")
    loader = SQuADLoader(config)

    logger.info("HuggingFace üzerinden veri seti indiriliyor ve Document formatında parse ediliyor...")
    try:
        documents = loader.load_documents()
        logger.info(f"Başarıyla {len(documents)} adet benzersiz (unique) paragraf Document olarak yüklendi.")
    except ImportError:
        logger.error("'datasets' kütüphanesi eksik. Lütfen 'uv add datasets' komutunu çalıştırın.")
        sys.exit(1)

    logger.info("Query (Soru) nesneleri ve Ground Truth haritası oluşturuluyor...")
    queries, ground_truth = loader.load()
    logger.info(f"Başarıyla {len(queries)} adet soru Query nesnesi olarak yüklendi.")

    # Çıktıları doğrulamak için bir klasör oluştur ve örnek kaydet
    output_dir = Path("experiments/data/squad_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dokümanları JSONL olarak kaydet
    docs_file = output_dir / "documents.jsonl"
    with open(docs_file, "w", encoding="utf-8") as f:
        for doc in documents:
            doc_dict = {
                "id": doc.id,
                "content": doc.content,
                "metadata": {
                    "source": doc.metadata.source,
                    "custom": doc.metadata.custom
                }
            }
            f.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")
    logger.info(f"Tüm dokümanlar {docs_file} konumuna kaydedildi.")

    # 2. Query ve Ground Truth'u kaydet
    queries_file = output_dir / "queries.jsonl"
    with open(queries_file, "w", encoding="utf-8") as f:
        for query in queries:
            query_dict = {
                "id": query.id,
                "text": query.text,
                "ground_truth_context_ids": list(ground_truth.get(query.id, set()))
            }
            f.write(json.dumps(query_dict, ensure_ascii=False) + "\n")
    logger.info(f"Tüm sorular ve hedefler (ground truth) {queries_file} konumuna kaydedildi.")

    logger.info("✅ SQuAD v2 parse işlemi nesnel performans ölçümü için tamamlandı!")

if __name__ == "__main__":
    main()
