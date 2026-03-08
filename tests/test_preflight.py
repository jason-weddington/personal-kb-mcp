"""Tests for CWD-based project context injection (preflight)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from personal_kb.preflight import (
    _format_expiry_badge,
    _get_known_projects,
    build_preflight_context,
    detect_project,
)

# ---------------------------------------------------------------------------
# detect_project — fuzzy matching
# ---------------------------------------------------------------------------


class TestDetectProject:
    """Fuzzy matching of CWD leaf directory to project_refs."""

    def test_exact_match(self) -> None:
        assert detect_project("/home/user/personal-kb", ["personal-kb"]) == ["personal-kb"]

    def test_separator_normalization(self) -> None:
        """personal_kb directory matches personal-kb project_ref."""
        assert detect_project("/home/user/personal_kb", ["personal-kb"]) == ["personal-kb"]

    def test_case_insensitive(self) -> None:
        assert detect_project("/home/user/PersonalKB", ["personal-kb"]) == ["personal-kb"]

    def test_mixed_separators_and_case(self) -> None:
        assert detect_project("/home/user/Personal_KB", ["personal-kb"]) == ["personal-kb"]

    def test_no_match(self) -> None:
        assert detect_project("/home/user/unrelated-project", ["personal-kb"]) == []

    def test_short_substring_rejected(self) -> None:
        """'kb' alone should NOT match 'personal-kb' (length ratio too low)."""
        assert detect_project("/home/user/kb", ["personal-kb"]) == []

    def test_substring_match_with_good_ratio(self) -> None:
        """'personal-kb' matches 'personal-kb-v2' (ratio 10/14 > 0.5)."""
        assert detect_project("/home/user/personal-kb", ["personal-kb-v2"]) == ["personal-kb-v2"]

    def test_substring_match_reverse(self) -> None:
        """'personal-kb-v2' directory matches 'personal-kb' project."""
        assert detect_project("/home/user/personal-kb-v2", ["personal-kb"]) == ["personal-kb"]

    def test_fuzzy_typo(self) -> None:
        """Minor typo caught by SequenceMatcher."""
        assert detect_project("/home/user/personl-kb", ["personal-kb"]) == ["personal-kb"]

    def test_multiple_matches(self) -> None:
        """CWD can match multiple projects."""
        projects = ["personal-kb", "personal-kb-team"]
        matches = detect_project("/home/user/personal-kb", projects)
        assert "personal-kb" in matches
        assert "personal-kb-team" in matches

    def test_empty_cwd(self) -> None:
        assert detect_project("", ["personal-kb"]) == []

    def test_root_cwd(self) -> None:
        assert detect_project("/", ["personal-kb"]) == []

    def test_no_known_projects(self) -> None:
        assert detect_project("/home/user/personal-kb", []) == []

    def test_dot_separator(self) -> None:
        """Dots treated as separators."""
        assert detect_project("/home/user/personal.kb", ["personal-kb"]) == ["personal-kb"]

    def test_trailing_slash(self) -> None:
        """Trailing slash doesn't break leaf extraction."""
        assert detect_project("/home/user/personal-kb/", ["personal-kb"]) == ["personal-kb"]

    def test_very_short_dissimilar_rejected(self) -> None:
        """Short dissimilar strings rejected even with substring check."""
        assert detect_project("/home/user/zz", ["personal-kb"]) == []

    def test_three_char_substring_accepted(self) -> None:
        """Three-char substring with good ratio accepted."""
        assert detect_project("/home/user/abc", ["abcd"]) == ["abcd"]

    def test_very_short_similar_matches_via_fuzzy(self) -> None:
        """Short similar strings can match via SequenceMatcher (bias toward recall)."""
        # "ab" vs "abc" has ratio 0.8 >= 0.75
        assert detect_project("/home/user/ab", ["abc"]) == ["abc"]


# ---------------------------------------------------------------------------
# _format_expiry_badge
# ---------------------------------------------------------------------------


class TestFormatExpiryBadge:
    def test_expired_days_ago(self) -> None:
        expired = (datetime.now(UTC) - timedelta(days=3, hours=1)).isoformat()
        assert _format_expiry_badge(expired) == " [EXPIRED 3d ago]"

    def test_expired_today(self) -> None:
        expired = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert _format_expiry_badge(expired) == " [EXPIRED today]"

    def test_expires_in_days(self) -> None:
        # Add extra hours to avoid rounding down across day boundary
        future = (datetime.now(UTC) + timedelta(days=5, hours=1)).isoformat()
        assert _format_expiry_badge(future) == " [EXPIRES 5d]"

    def test_expires_in_hours(self) -> None:
        # Use half-hours to avoid boundary rounding
        future = (datetime.now(UTC) + timedelta(hours=8, minutes=30)).isoformat()
        assert _format_expiry_badge(future) == " [EXPIRES 8h]"

    def test_invalid_string(self) -> None:
        assert _format_expiry_badge("not-a-date") == ""


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def preflight_db():
    """Create a SQLite DB with test entries for preflight queries."""
    from personal_kb.db.connection import create_connection

    db = await create_connection(":memory:", embedding_dim=64)

    now = datetime.now(UTC)

    # Insert test entries for project "my-project"
    entries = [
        # Expiring entry (expires in 5 days)
        (
            "kb-00001",
            "my-project",
            "API rate limit",
            "API rate limit for v2",
            "Rate limit is 100 req/min",
            "factual_reference",
            now.isoformat(),
            now.isoformat(),
            (now + timedelta(days=5)).isoformat(),
        ),
        # Recently expired (2 days ago)
        (
            "kb-00002",
            "my-project",
            "Sprint goal",
            "Sprint 42 goal",
            "Ship preflight feature",
            "factual_reference",
            now.isoformat(),
            now.isoformat(),
            (now - timedelta(days=2)).isoformat(),
        ),
        # Decision (recent)
        (
            "kb-00003",
            "my-project",
            "Chose SQLite",
            "Chose SQLite over Postgres for MVP",
            "SQLite is simpler for single-user",
            "decision",
            now.isoformat(),
            now.isoformat(),
            None,
        ),
        # Lesson learned (recent)
        (
            "kb-00004",
            "my-project",
            "FTS5 gotcha",
            "FTS5 requires content sync triggers",
            "Without triggers, FTS index gets stale",
            "lesson_learned",
            now.isoformat(),
            now.isoformat(),
            None,
        ),
        # Convention
        (
            "kb-00005",
            "my-project",
            "Conventional commits",
            "Use conventional commit format",
            "feat: fix: chore: docs:",
            "pattern_convention",
            now.isoformat(),
            now.isoformat(),
            None,
        ),
        # Different project (should not appear)
        (
            "kb-00006",
            "other-project",
            "Unrelated",
            "Unrelated entry",
            "Should not appear",
            "decision",
            now.isoformat(),
            now.isoformat(),
            None,
        ),
        # Expired beyond grace window (8 days ago, should not appear)
        (
            "kb-00007",
            "my-project",
            "Old expired",
            "Old expired entry",
            "Way past grace window",
            "factual_reference",
            now.isoformat(),
            now.isoformat(),
            (now - timedelta(days=8)).isoformat(),
        ),
        # Deactivated entry (should not appear)
        (
            "kb-00008",
            "my-project",
            "Deactivated",
            "Deactivated entry",
            "Should not appear",
            "decision",
            now.isoformat(),
            now.isoformat(),
            None,
        ),
    ]

    for e in entries:
        is_active = 0 if e[0] == "kb-00008" else 1
        await db.execute(
            "INSERT INTO knowledge_entries "
            "(id, project_ref, short_title, long_title, knowledge_details, "
            "entry_type, created_at, updated_at, expires_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [*e, is_active],
        )
    await db.commit()

    yield db
    await db.close()


class TestGetKnownProjects:
    async def test_returns_distinct_projects(self, preflight_db) -> None:
        projects = await _get_known_projects(preflight_db)
        assert sorted(projects) == ["my-project", "other-project"]


class TestBuildPreflightContext:
    async def test_matching_project(self, preflight_db) -> None:
        """CWD 'my_project' matches 'my-project' and returns ToC."""
        result = await build_preflight_context(preflight_db, "/home/user/my_project")
        assert "# Project Context" in result
        assert "my-project" in result
        # Expiring entries
        assert "kb-00001" in result
        assert "EXPIRES" in result
        # Recently expired
        assert "kb-00002" in result
        assert "EXPIRED" in result
        # Decision
        assert "kb-00003" in result
        # Lesson
        assert "kb-00004" in result
        # Convention
        assert "kb-00005" in result
        # Should NOT include
        assert "kb-00006" not in result  # different project
        assert "kb-00007" not in result  # beyond grace window
        assert "kb-00008" not in result  # deactivated

    async def test_no_matching_project(self, preflight_db) -> None:
        result = await build_preflight_context(preflight_db, "/home/user/unrelated")
        assert result == ""

    async def test_section_structure(self, preflight_db) -> None:
        result = await build_preflight_context(preflight_db, "/home/user/my_project")
        assert "Expiring:" in result
        assert "Recent decisions & lessons:" in result
        assert "Conventions:" in result
        assert "kb_get" in result  # instruction to use kb_get

    async def test_other_project(self, preflight_db) -> None:
        """Exact match for other-project returns only its entries."""
        result = await build_preflight_context(preflight_db, "/home/user/other-project")
        assert "kb-00006" in result
        assert "kb-00001" not in result

    async def test_entry_count_in_header(self, preflight_db) -> None:
        result = await build_preflight_context(preflight_db, "/home/user/my_project")
        # 5 entries: 2 expiring + 2 recent + 1 convention
        assert "5 entries" in result


# ---------------------------------------------------------------------------
# Team filtering
# ---------------------------------------------------------------------------


@pytest.fixture()
async def team_db():
    """DB with entries across multiple teams for the same project."""
    from personal_kb.db.connection import create_connection

    db = await create_connection(":memory:", embedding_dim=64)

    now = datetime.now(UTC)

    entries = [
        # team-alpha decision
        (
            "kb-00010",
            "shared-proj",
            "Alpha decision",
            "Alpha chose X",
            "details",
            "decision",
            "team-alpha",
        ),
        # team-beta decision
        (
            "kb-00011",
            "shared-proj",
            "Beta decision",
            "Beta chose Y",
            "details",
            "decision",
            "team-beta",
        ),
        # No-team decision (global)
        (
            "kb-00012",
            "shared-proj",
            "Global decision",
            "Everyone agreed on Z",
            "details",
            "decision",
            None,
        ),
        # team-alpha convention
        (
            "kb-00013",
            "shared-proj",
            "Alpha style",
            "Alpha code style",
            "details",
            "pattern_convention",
            "team-alpha",
        ),
        # team-beta convention
        (
            "kb-00014",
            "shared-proj",
            "Beta style",
            "Beta code style",
            "details",
            "pattern_convention",
            "team-beta",
        ),
        # team-alpha expiring
        (
            "kb-00015",
            "shared-proj",
            "Alpha deadline",
            "Alpha sprint deadline",
            "details",
            "factual_reference",
            "team-alpha",
        ),
    ]

    for e in entries:
        expires_at = (now + timedelta(days=3)).isoformat() if e[0] == "kb-00015" else None
        await db.execute(
            "INSERT INTO knowledge_entries "
            "(id, project_ref, short_title, long_title, knowledge_details, "
            "entry_type, team, created_at, updated_at, expires_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            [*e, now.isoformat(), now.isoformat(), expires_at],
        )
    await db.commit()

    yield db
    await db.close()


class TestTeamFiltering:
    async def test_no_team_returns_all(self, team_db) -> None:
        """Without team filter, all entries from all teams appear."""
        result = await build_preflight_context(team_db, "/home/user/shared-proj")
        assert "kb-00010" in result  # alpha
        assert "kb-00011" in result  # beta
        assert "kb-00012" in result  # global

    async def test_team_alpha_sees_own_and_global(self, team_db) -> None:
        """team-alpha sees alpha entries + global, not beta."""
        result = await build_preflight_context(team_db, "/home/user/shared-proj", team="team-alpha")
        assert "kb-00010" in result  # alpha decision
        assert "kb-00012" in result  # global decision
        assert "kb-00013" in result  # alpha convention
        assert "kb-00015" in result  # alpha expiring
        assert "kb-00011" not in result  # beta decision
        assert "kb-00014" not in result  # beta convention

    async def test_team_beta_sees_own_and_global(self, team_db) -> None:
        """team-beta sees beta entries + global, not alpha."""
        result = await build_preflight_context(team_db, "/home/user/shared-proj", team="team-beta")
        assert "kb-00011" in result  # beta decision
        assert "kb-00012" in result  # global decision
        assert "kb-00014" in result  # beta convention
        assert "kb-00010" not in result  # alpha decision
        assert "kb-00013" not in result  # alpha convention
        assert "kb-00015" not in result  # alpha expiring
