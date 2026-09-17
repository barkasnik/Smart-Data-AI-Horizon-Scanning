from pathlib import Path


def test_dashboard_exists_and_loads_weekly_monthly_json():
    root = Path(__file__).resolve().parents[1]
    text = (root / "index.html").read_text(encoding="utf-8")
    assert "Smart Data &amp; AI Horizon Scanner" in text
    assert "output/${mode}-latest.json" in text
    assert "PESTLE" in text
    assert "SWOT" in text
    assert "Open original article" in text


def test_workflow_is_in_github_workflows_directory():
    root = Path(__file__).resolve().parents[1]
    workflow = root / ".github" / "workflows" / "radar.yml"
    assert workflow.exists()
    text = workflow.read_text(encoding="utf-8")
    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "weekly" in text
    assert "monthly" in text
    assert "qwen2.5:3b-instruct" in text
