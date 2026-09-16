from smart_data_radar.dedupe import dedupe_candidates
from smart_data_radar.models import Candidate


def test_tracking_parameters_are_deduped():
    items = [
        Candidate(url="https://example.com/a?utm_source=x", title="A"),
        Candidate(url="https://example.com/a", title="A again"),
    ]
    assert len(dedupe_candidates(items)) == 1

