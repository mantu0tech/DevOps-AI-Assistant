from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os


llm = OllamaLLM(model="llama3")

docs = []
for file in os.listdir("devops_data"):
    loader = TextLoader(f"devops_data/{file}")
    docs.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Store in vector DB
db = Chroma.from_documents(
    documents,
    embedding=OllamaEmbeddings(model="llama3")
)

SYSTEM_PROMPT = """
I am your Devops Expert..
Answer ONLY questions related to:
DevOps, Cloud, SRE, CI/CD, Linux, Containers, Kubernetes, Docker,
Terraform, Ansible, Jenkins, Git, AWS, Azure, GCP, Monitoring, Networking, Grafana, Prometheous, ELK, YAML
and Infrastructure-related questions.
If not related, say:
'I am a DevOps-focused assistant and can only help with DevOps-related questions.'
"""

while True:
    query = input("Ask anything related DevOps: ")
    if query.lower() == "exit":
        break

    docs = db.similarity_search(query)
    context = "\n".join([d.page_content for d in docs])

    response = llm.invoke(SYSTEM_PROMPT + context + query)
    print("Bot:", response)

