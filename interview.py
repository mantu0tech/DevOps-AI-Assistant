from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
import os

# ----------------------------
# CONFIG
# ----------------------------
TOTAL_QUESTIONS = 5
PASS_SCORE = 3
CACHE_DIR = "chroma_db"  # folder to store embeddings

# ----------------------------
# LLM SETUP
# ----------------------------
llm = OllamaLLM(model="llama3")

# ----------------------------
# INTERVIEW LINKS (FREE)
# ----------------------------
URLS = [
    "https://www.interviewbit.com/devops-interview-questions/",
    "https://www.interviewbit.com/linux-interview-questions/",
    "https://www.interviewbit.com/aws-interview-questions/",
    "https://www.interviewbit.com/ansible-interview-questions/",
    "https://www.interviewbit.com/terraform-interview-questions/",
    "https://www.interviewbit.com/monitoring-interview-questions/",
    "https://www.interviewbit.com/docker-interview-questions/",
    "https://www.interviewbit.com/kubernetes-interview-questions/"
]

# ----------------------------
# LOAD & STORE QUESTIONS
# ----------------------------
if os.path.exists(CACHE_DIR):
    print("Loading cached embeddings...")
    db = Chroma(persist_directory=CACHE_DIR, embedding_function=OllamaEmbeddings(model="llama3"))
    documents = []  # Optional: can't access original docs from cache
else:
    print("Loading interview questions from web ...")
    loader = WebBaseLoader(URLS)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    documents = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents,
        embedding=OllamaEmbeddings(model="llama3"),
        persist_directory=CACHE_DIR
    )
    db.persist()

print("Sample chunks loaded:")
if documents:
    for doc in documents[:5]:
        print(doc.page_content[:200])
else:
    print("⚠️ Original documents not loaded (cache only). Similarity search will still work.")

# ----------------------------
# SYSTEM PROMPT
# ----------------------------
SYSTEM_PROMPT = """
You are a senior DevOps interviewer.
Evaluate the candidate answer.
If the answer is mostly correct, say "CORRECT".
If the answer is wrong or incomplete, say "INCORRECT".
Then give brief feedback and the correct answer.
"""

# ----------------------------
# INTERVIEW SESSION
# ----------------------------
topic = input("Choose topic (devops/linux/aws/ansible/terraform/monitoring/docker/kubernetes): ").lower()
score = 0

for i in range(1, TOTAL_QUESTIONS + 1):
    print(f"\n📌 Question {i}/{TOTAL_QUESTIONS}")
    result = db.similarity_search(topic, k=1)
    if not result:
        print("⚠️ No question found for this topic. Skipping...")
        continue

    question = result[0].page_content
    print("\n🧑‍💼 Interviewer:\n")
    print(question[:500])

    user_answer = input("\n👤 Your Answer: ")

    evaluation = llm.invoke(
        SYSTEM_PROMPT +
        "\nQuestion:\n" + question +
        "\nCandidate Answer:\n" + user_answer
    )

    print("\n📋 Feedback:\n")
    print(evaluation)

    if "CORRECT" in evaluation.upper():
        score += 1

# ----------------------------
# FINAL RESULT
# ----------------------------
print("\n" + "=" * 40)
print("🏁 INTERVIEW RESULT")
print("=" * 40)
print(f"Total Questions : {TOTAL_QUESTIONS}")
print(f"Correct Answers : {score}")
print(f"Incorrect       : {TOTAL_QUESTIONS - score}")
print("✅ Status: PASSED" if score >= PASS_SCORE else "❌ Status: FAILED")
