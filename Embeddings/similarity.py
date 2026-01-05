from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embed_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  
)

document=[
    "virat Kohli is a great cricketer",
    "Sachin Tendulkar is a legendary batsman",
    "Rohit Sharma is known for his big hundreds",
    "Bumrah is a top fast bowler",
    "MS Dhoni is one of the best wicketkeepers"
]

document_embedding=embed_model.embed_documents(document)
query_embedding=embed_model.embed_query("Who is the best bowler?")

results=cosine_similarity([query_embedding], document_embedding)[0]
ans=sorted(list(enumerate(results)),key=lambda x:x[1])
print(document[ans[-1][0]])


