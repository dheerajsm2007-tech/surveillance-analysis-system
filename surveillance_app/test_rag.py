import sys
sys.path.insert(0, ".")
from part2.chain import ask

VIDEO = "test_surveillance.mp4"
QUESTIONS = [
    "How many people were detected in this video?",
    "Who had the longest dwell time, and how long was it?",
    "Were there any anomalies or suspicious activity?",
    "Which zones were visited the most?",
    "What happened near the exit door?",
    "Was any vehicle detected?",
]

SEP = "=" * 64
print(SEP)
print("RAG Q&A TEST  —  test_surveillance.mp4")
print(SEP)

for i, q in enumerate(QUESTIONS, 1):
    print(f"\nQ{i}: {q}")
    print("-" * 50)
    result = ask(q, VIDEO)
    print(f"ANSWER: {result['answer']}")
    print(f"SOURCES ({len(result['sources'])}):")
    for s in result["sources"]:
        label = s["label"]
        tid   = s["track_id"]
        sim   = s["similarity"]
        text  = s["text"]
        print(f"  [{label} #{tid:03d}  sim={sim}]  {text}")

print()
print(SEP)
print("RAG TEST COMPLETE")
print(SEP)
