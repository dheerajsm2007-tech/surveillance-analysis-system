import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'c:/Users/Dheer/Downloads/locateanything_project/surveillance_app')

from part2.embedder import embed_events_for_video
from part2.chain import ask

VIDEO = "test_people.mp4"

print(f"Embedding events for {VIDEO}...")
n = embed_events_for_video(VIDEO)
print(f"Embedded {n} events.\n")

QUESTIONS = [
    "How many people were detected in this video?",
    "Which zones did the detected person visit?",
    "Were there any anomalies or suspicious activity?",
    "What was the peak occupancy?",
    "How long did the person stay?",
]

print("=" * 60)
print("RAG Q&A TEST")
print("=" * 60)
for q in QUESTIONS:
    print(f"\nQ: {q}")
    try:
        result = ask(q, VIDEO)
        print(f"A: {result['answer']}")
        if result['sources']:
            print(f"   (similarity: {result['sources'][0]['similarity']})")
    except Exception as e:
        print(f"ERROR: {e}")
print("\n" + "=" * 60)
