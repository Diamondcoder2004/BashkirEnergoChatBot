# Башкирэнерго RAG

```bash
docker build -t bashkir-rag .
docker run --gpus all -v ./documents:/app/documents -v ./output_md:/app/output_md bashkir-rag