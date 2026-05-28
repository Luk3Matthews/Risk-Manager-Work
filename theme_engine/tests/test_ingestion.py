"""Tests for Module 2: Ingestion — evidence scoring & theme management."""
from datetime import date, timedelta

import pytest

from theme_engine.config import get_config
from theme_engine.ingestion import (
    add_evidence_to_theme,
    build_active_ledger,
    compute_theme_confidence,
    compute_theme_strength,
    score_evidence,
)
from theme_engine.models import (
    Direction,
    EvidenceItem,
    Horizon,
    Theme,
    ThemeCategory,
    ThemeStatus,
)


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def sample_evidence():
    today = date.today()
    return [
        EvidenceItem(
            source="reuters", date=today,
            title="Test article 1",
            credibility_score=0.9, timeliness_score=0.8,
            corroboration_count=3,
        ),
        EvidenceItem(
            source="ft", date=today - timedelta(days=5),
            title="Test article 2",
            credibility_score=0.7, timeliness_score=0.6,
            corroboration_count=1,
        ),
        EvidenceItem(
            source="blog", date=today - timedelta(days=30),
            title="Old article",
            credibility_score=0.3, timeliness_score=0.2,
            corroboration_count=0,
        ),
    ]


@pytest.fixture
def sample_theme(sample_evidence):
    return Theme(
        theme_id="test_01",
        name="Test geopolitical theme",
        category=ThemeCategory.GEOPOLITICAL,
        direction=Direction.BEARISH,
        horizon=Horizon.NEAR_TERM,
        likelihood=0.5,
        status=ThemeStatus.ACTIVE,
        evidence=sample_evidence,
    )


class TestEvidenceScoring:
    def test_usefulness_formula(self, cfg):
        """U = 0.4*cred + 0.3*time + 0.3*min(corr/3, 1)."""
        item = EvidenceItem(
            source="test", date=date.today(), title="Test",
            credibility_score=1.0, timeliness_score=1.0,
            corroboration_count=3,
        )
        u = score_evidence(item, cfg)
        expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * min(3 / 3, 1.0)
        assert abs(u - expected) < 1e-10
        assert abs(u - 1.0) < 1e-10

    def test_usefulness_zero_corroboration(self, cfg):
        item = EvidenceItem(
            source="test", date=date.today(), title="Test",
            credibility_score=0.5, timeliness_score=0.5,
            corroboration_count=0,
        )
        u = score_evidence(item, cfg)
        expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.0
        assert abs(u - expected) < 1e-10

    def test_usefulness_saturates_corroboration(self, cfg):
        item = EvidenceItem(
            source="test", date=date.today(), title="Test",
            credibility_score=0.5, timeliness_score=0.5,
            corroboration_count=100,
        )
        u = score_evidence(item, cfg)
        expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 1.0  # min(100/3,1) = 1
        assert abs(u - expected) < 1e-10

    def test_usefulness_bounds(self, cfg):
        for cred in [0.0, 0.5, 1.0]:
            for time in [0.0, 0.5, 1.0]:
                for corr in [0, 2, 5]:
                    item = EvidenceItem(
                        source="test", date=date.today(), title="Test",
                        credibility_score=cred, timeliness_score=time,
                        corroboration_count=corr,
                    )
                    u = score_evidence(item, cfg)
                    assert 0.0 <= u <= 1.0


class TestThemeConfidence:
    def test_no_evidence(self, cfg):
        theme = Theme(
            name="Empty", category=ThemeCategory.GROWTH,
            status=ThemeStatus.ACTIVE,
        )
        c = compute_theme_confidence(theme, cfg=cfg)
        assert c == 0.0

    def test_single_perfect_evidence(self, cfg):
        """One perfect evidence item today → confidence close to usefulness."""
        item = EvidenceItem(
            source="test", date=date.today(), title="Test",
            credibility_score=1.0, timeliness_score=1.0,
            corroboration_count=3,
        )
        theme = Theme(
            name="Test", category=ThemeCategory.GROWTH,
            evidence=[item], status=ThemeStatus.ACTIVE,
        )
        c = compute_theme_confidence(theme, cfg=cfg)
        assert c > 0.9  # Near 1.0 for perfect evidence

    def test_stale_evidence_decays(self, cfg):
        """Old evidence should produce lower confidence than fresh."""
        today = date.today()
        fresh = EvidenceItem(
            source="test", date=today, title="Fresh",
            credibility_score=0.8, timeliness_score=0.8,
            corroboration_count=2,
        )
        stale = EvidenceItem(
            source="test", date=today - timedelta(days=90), title="Stale",
            credibility_score=0.8, timeliness_score=0.8,
            corroboration_count=2,
        )

        t_fresh = Theme(name="Fresh", category=ThemeCategory.GROWTH,
                        evidence=[fresh], status=ThemeStatus.ACTIVE)
        t_stale = Theme(name="Stale", category=ThemeCategory.GROWTH,
                        evidence=[stale], status=ThemeStatus.ACTIVE)

        c_fresh = compute_theme_confidence(t_fresh, cfg=cfg)
        c_stale = compute_theme_confidence(t_stale, cfg=cfg)
        assert c_fresh > c_stale

    def test_more_evidence_increases_confidence(self, cfg):
        today = date.today()
        items = [
            EvidenceItem(
                source="test", date=today - timedelta(days=i), title=f"Item {i}",
                credibility_score=0.7, timeliness_score=0.7,
                corroboration_count=1,
            )
            for i in range(5)
        ]
        t1 = Theme(name="Few", category=ThemeCategory.GROWTH,
                    evidence=items[:1], status=ThemeStatus.ACTIVE)
        t5 = Theme(name="Many", category=ThemeCategory.GROWTH,
                    evidence=items, status=ThemeStatus.ACTIVE)

        c1 = compute_theme_confidence(t1, cfg=cfg)
        c5 = compute_theme_confidence(t5, cfg=cfg)
        assert c5 > c1


class TestThemeStrength:
    def test_strength_formula(self, cfg):
        """S = likelihood^0.5 * confidence^0.5."""
        theme = Theme(
            name="Test", category=ThemeCategory.GROWTH,
            likelihood=0.64, confidence=0.81,
            status=ThemeStatus.ACTIVE,
        )
        s = compute_theme_strength(theme, cfg=cfg)
        expected = 0.64 ** 0.5 * 0.81 ** 0.5
        assert abs(s - expected) < 1e-6

    def test_strength_bounds(self, cfg):
        for lik in [0.0, 0.3, 0.7, 1.0]:
            for conf in [0.0, 0.3, 0.7, 1.0]:
                theme = Theme(
                    name="Test", category=ThemeCategory.GROWTH,
                    likelihood=lik, confidence=conf,
                    status=ThemeStatus.ACTIVE,
                )
                s = compute_theme_strength(theme, cfg=cfg)
                assert 0.0 <= s <= 1.0


class TestActiveLedger:
    def test_filters_retired(self, cfg):
        themes = [
            Theme(name="Active", category=ThemeCategory.GROWTH,
                  status=ThemeStatus.ACTIVE, likelihood=0.5,
                  evidence=[EvidenceItem(source="x", date=date.today(),
                            title="X", credibility_score=0.5,
                            timeliness_score=0.5, corroboration_count=1)]),
            Theme(name="Retired", category=ThemeCategory.GROWTH,
                  status=ThemeStatus.RETIRED, likelihood=0.8,
                  evidence=[EvidenceItem(source="x", date=date.today(),
                            title="X", credibility_score=0.9,
                            timeliness_score=0.9, corroboration_count=3)]),
        ]
        ledger = build_active_ledger(themes, cfg=cfg)
        assert len(ledger) == 1
        assert ledger[0].name == "Active"

    def test_sorts_by_strength(self, cfg):
        today = date.today()
        themes = [
            Theme(name="Weak", category=ThemeCategory.GROWTH,
                  status=ThemeStatus.ACTIVE, likelihood=0.2,
                  evidence=[EvidenceItem(source="x", date=today, title="X",
                            credibility_score=0.3, timeliness_score=0.3,
                            corroboration_count=0)]),
            Theme(name="Strong", category=ThemeCategory.GROWTH,
                  status=ThemeStatus.ACTIVE, likelihood=0.9,
                  evidence=[EvidenceItem(source="x", date=today, title="X",
                            credibility_score=0.95, timeliness_score=0.95,
                            corroboration_count=5)]),
        ]
        ledger = build_active_ledger(themes, cfg=cfg)
        assert ledger[0].name == "Strong"

    def test_respects_max_themes(self, cfg):
        today = date.today()
        themes = [
            Theme(name=f"T{i}", category=ThemeCategory.GROWTH,
                  status=ThemeStatus.ACTIVE, likelihood=0.5,
                  evidence=[EvidenceItem(source="x", date=today, title="X",
                            credibility_score=0.5, timeliness_score=0.5,
                            corroboration_count=1)])
            for i in range(30)
        ]
        ledger = build_active_ledger(themes, max_themes=5, cfg=cfg)
        assert len(ledger) == 5
