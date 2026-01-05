from langchain_huggingface import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

vector=embed_model.embed_query("Hello world")
print("Embedding vector:", str(vector))