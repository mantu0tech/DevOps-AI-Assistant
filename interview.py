from langchain_ollama import OllamaLLM
import random

# ----------------------------
# CONFIG
# ----------------------------
TOTAL_QUESTIONS = 5
PASS_SCORE = 3

# ----------------------------
# LLM SETUP
# ----------------------------
print("🔧 Initializing LLM...")
llm = OllamaLLM(model="llama3")

# ----------------------------
# TOPICS
# ----------------------------
TOPICS = {
    "devops": "DevOps practices, CI/CD, automation, and culture",
    "linux": "Linux commands, system administration, shell scripting",
    "aws": "AWS services, cloud architecture, EC2, S3, Lambda",
    "ansible": "Ansible playbooks, automation, configuration management",
    "terraform": "Infrastructure as Code, Terraform syntax, state management",
    "monitoring": "Monitoring tools, metrics, logging, observability",
    "docker": "Docker containers, images, dockerfile, docker-compose",
    "kubernetes": "K8s architecture, pods, services, deployments"
}

# ----------------------------
# TOPIC SELECTION
# ----------------------------
print("\n" + "="*60)
print("🎯 DEVOPS INTERVIEW SIMULATOR - AI POWERED")
print("="*60)
print("\nAvailable Topics:")
for idx, (topic, desc) in enumerate(TOPICS.items(), 1):
    print(f"  {idx}. {topic.upper():<15} - {desc}")

topic = input("\n👉 Choose topic: ").lower().strip()

if topic not in TOPICS:
    print(f"❌ Invalid topic. Choose from: {', '.join(TOPICS.keys())}")
    exit()

difficulty = input("👉 Choose difficulty (beginner/intermediate/advanced): ").lower().strip()
if difficulty not in ['beginner', 'intermediate', 'advanced']:
    difficulty = 'intermediate'

print(f"\n✅ Starting {topic.upper()} interview ({difficulty} level)")
print("⏳ Generating {0} unique questions...\n".format(TOTAL_QUESTIONS))

# ----------------------------
# GENERATE QUESTIONS
# ----------------------------
QUESTION_PROMPT = f"""You are a senior DevOps interviewer. Generate {TOTAL_QUESTIONS} unique {difficulty}-level interview questions about {topic} ({TOPICS[topic]}).

Requirements:
- Each question should be practical and commonly asked in real interviews
- Mix of theoretical and scenario-based questions
- Questions should be clear and specific
- Number each question (1., 2., 3., etc.)
- Only provide the questions, no answers

Format:
1. [First question]
2. [Second question]
3. [Third question]
etc.
"""

try:
    questions_text = llm.invoke(QUESTION_PROMPT)
    
    # Parse questions
    import re
    questions = re.findall(r'\d+\.\s+(.+?)(?=\d+\.|$)', questions_text, re.DOTALL)
    questions = [q.strip() for q in questions if len(q.strip()) > 10]
    
    if len(questions) < TOTAL_QUESTIONS:
        print(f"⚠️ Only generated {len(questions)} questions")
        TOTAL_QUESTIONS = len(questions)
    else:
        questions = questions[:TOTAL_QUESTIONS]
    
except Exception as e:
    print(f"❌ Error generating questions: {e}")
    exit()

# ----------------------------
# EVALUATION PROMPT
# ----------------------------
EVAL_PROMPT = """You are a senior DevOps interviewer evaluating a candidate's answer.

Evaluate based on:
1. Technical accuracy
2. Completeness of answer
3. Practical understanding
4. Real-world applicability

Response format:
- Start with either "CORRECT" or "INCORRECT"
- Provide 2-3 sentences of constructive feedback
- If incorrect, briefly mention key points they missed

Be fair but thorough in your evaluation."""

# ----------------------------
# INTERVIEW SESSION
# ----------------------------
print("="*60)
input("Press ENTER when ready to start...")

score = 0
answers_log = []

for i in range(TOTAL_QUESTIONS):
    print(f"\n{'='*60}")
    print(f"📌 QUESTION {i+1}/{TOTAL_QUESTIONS}")
    print('='*60)
    
    question = questions[i]
    print(f"\n🧑‍💼 {question}\n")
    
    user_answer = input("👤 YOUR ANSWER: ").strip()
    
    if not user_answer:
        print("\n❌ No answer provided - marking as incorrect\n")
        answers_log.append({
            'question': question,
            'answer': user_answer,
            'correct': False,
            'feedback': 'No answer provided'
        })
        continue
    
    print("\n⏳ Evaluating your answer...\n")
    
    try:
        evaluation = llm.invoke(
            EVAL_PROMPT +
            f"\n\nQUESTION: {question}" +
            f"\n\nCANDIDATE'S ANSWER: {user_answer}" +
            f"\n\nYOUR EVALUATION:"
        )
        
        print("📋 FEEDBACK:")
        print("-" * 60)
        print(evaluation)
        print("-" * 60)
        
        is_correct = "CORRECT" in evaluation.upper()
        
        if is_correct:
            score += 1
            print("\n✅ +1 Point!")
        else:
            print("\n❌ No points")
        
        answers_log.append({
            'question': question,
            'answer': user_answer,
            'correct': is_correct,
            'feedback': evaluation
        })
            
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        continue
    
    if i < TOTAL_QUESTIONS - 1:
        input("\nPress ENTER for next question...")

# ----------------------------
# FINAL RESULT
# ----------------------------
percentage = (score / TOTAL_QUESTIONS) * 100

print("\n" + "=" * 60)
print("🏁 FINAL RESULTS")
print("=" * 60)
print(f"📊 Score: {score}/{TOTAL_QUESTIONS} ({percentage:.0f}%)")
print(f"✅ Correct: {score}")
print(f"❌ Incorrect: {TOTAL_QUESTIONS - score}")
print("=" * 60)

if score >= PASS_SCORE:
    print("\n🎉 STATUS: PASSED! 🎉")
    print("Excellent work! You demonstrated strong knowledge.")
else:
    print(f"\n😔 STATUS: FAILED (needed {PASS_SCORE}/{TOTAL_QUESTIONS})")
    print("Keep studying and try again! Practice makes perfect.")

# ----------------------------
# REVIEW SUMMARY
# ----------------------------
print("\n" + "=" * 60)
print("📝 QUESTION REVIEW")
print("=" * 60)

for idx, log in enumerate(answers_log, 1):
    status = "✅" if log['correct'] else "❌"
    print(f"\n{status} Q{idx}: {log['question'][:80]}...")
    print(f"   Your answer: {log['answer'][:100]}...")

print("\n👋 Thanks for participating!")
print("\n💡 TIP: Run the script again for a fresh set of questions!\n")