"""Re-evaluate the unchanged retained run; exit 2 on its recorded quality failure."""
import json
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[2]))
from live import quality_verdict

summary = json.loads((here.parent / 'raw/summary.json').read_text(encoding='utf-8'))
outcome = {'assessment': 'retrospective; the original experiment predates this quality gate',
           'original_controller_exit': 0, 'measurement_status': 'complete', **quality_verdict(summary)}
(here / 'quality-verdict.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
print(json.dumps(outcome, indent=2))
sys.exit(0 if outcome['equal_quality_comparison_eligible'] else 2)
