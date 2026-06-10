import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'c:/Users/Dheer/Downloads/locateanything_project/eagle/Embodied')
sys.path.insert(0, 'c:/Users/Dheer/Downloads/locateanything_project/surveillance_app')

from part1.detector import Detector
from part1.pipeline import run_pipeline

d = Detector()
d.load()
print('Model loaded. Running pipeline on test_people.mp4...')

result = run_pipeline(
    r'c:\Users\Dheer\Downloads\locateanything_project\test_people.mp4',
    save_to_db=True,
    detector=d
)

print('\n=== PIPELINE RESULT ===')
print(result['summary_text'])
print(f"Persons detected: {result['total_persons']}")
print(f"Duration: {result['duration']:.1f}s")
print(f"Anomalies: {result['anomaly_count']}")
print(f"Events: {len(result['events'])}")
