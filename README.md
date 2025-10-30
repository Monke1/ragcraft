# ragcraft

End-to-end RAG pipeline with built-in evaluation. Ingest documents, chunk, embed, store, retrieve and generate answers. Includes metrics for MRR, NDCG, precision, recall and faithfulness.

## Install

```
pip install -e .
```

## Usage

```python
from ragcraft import Pipeline
from ragcraft.loaders import load_from_string

pipe = Pipeline()
doc = load_from_string("your document text here")
pipe.ingest([doc])
result = pipe.retrieve("your question")
print(result.chunks[0].text)
```

## License

MIT

