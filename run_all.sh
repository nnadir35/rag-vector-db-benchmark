echo "=== 1. grep outputs ==="
grep -n "top_k" experiments/configs/baseline_faiss.yaml experiments/configs/baseline_milvus.yaml experiments/configs/baseline_qdrant.yaml experiments/configs/baseline_weaviate.yaml

echo "=== 2. Archive results ==="
cd experiments/results
mkdir -p archive
mv official_baseline_faiss_100q_20260715_194936.json archive/archive_baseline_faiss_100q_topk3_superseded_20260715_194936.json 2>&1 || echo "FAISS missing"
mv official_baseline_qdrant_100q_20260715_201200.json archive/archive_baseline_qdrant_100q_topk3_superseded_20260715_201200.json 2>&1 || echo "QDRANT missing"
cd ../..

echo "=== 3. Run Experiments ==="
echo "=== FAISS (top_k=10) ===" && python scripts/run_experiment.py --config experiments/configs/baseline_faiss.yaml
echo "=== MILVUS (top_k=10) ===" && python scripts/run_experiment.py --config experiments/configs/baseline_milvus.yaml
echo "=== QDRANT (top_k=10) ===" && python scripts/run_experiment.py --config experiments/configs/baseline_qdrant.yaml
echo "=== WEAVIATE (top_k=10) ===" && python scripts/run_experiment.py --config experiments/configs/baseline_weaviate.yaml

echo "=== 4. Rename Files ==="
cd experiments/results
for pattern in baseline_faiss_squad baseline_milvus_squad baseline_qdrant_squad baseline_weaviate_squad; do
  latest=$(ls -t *${pattern}*.json | grep -v topk10 | head -1)
  if [ -n "$latest" ]; then
    timestamp=$(echo $latest | grep -oE '[0-9]{8}_[0-9]{6}')
    db_name=$(echo $pattern | cut -d'_' -f2)
    new_name="official_baseline_${db_name}_100q_topk10_${timestamp}.json"
    mv "$latest" "$new_name"
    echo "$latest -> $new_name"
  fi
done
cd ../..

echo "=== 5. Summary Table ==="
python3 -c "
import json, glob
for f in sorted(glob.glob('experiments/results/official_baseline_*_topk10_*.json')):
    try:
        d = json.load(open(f))
        print(f.split('/')[-1], '->', d.get('metrics', d.get('metrics_summary', {})))
    except Exception as e:
        print(f, 'Error reading:', e)
"
