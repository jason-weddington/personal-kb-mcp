"""Tests for project context primer (preflight)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from personal_kb.preflight import (
    _format_expiry_badge,
    build_project_context,
)

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


class TestBuildProjectContext:
    async def test_matching_project(self, preflight_db) -> None:
        """Explicit project_ref returns ToC for that project."""
        result = await build_project_context(preflight_db, "my-project")
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

    async def test_no_entries_for_project(self, preflight_db) -> None:
        result = await build_project_context(preflight_db, "nonexistent")
        assert "No entries found" in result

    async def test_section_structure(self, preflight_db) -> None:
        result = await build_project_context(preflight_db, "my-project")
        assert "Expiring:" in result
        assert "Recent decisions & lessons:" in result
        assert "Conventions:" in result
        assert "kb_get" in result  # instruction to use kb_get

    async def test_other_project(self, preflight_db) -> None:
        """other-project returns only its entries."""
        result = await build_project_context(preflight_db, "other-project")
        assert "kb-00006" in result
        assert "kb-00001" not in result

    async def test_entry_count_in_header(self, preflight_db) -> None:
        result = await build_project_context(preflight_db, "my-project")
        # 5 entries: 2 expiring + 2 recent + 1 convention
        assert "5 entries" in result


# ---------------------------------------------------------------------------
# since parameter
# ---------------------------------------------------------------------------


@pytest.fixture()
async def since_db():
    """DB with entries at different ages for testing since filter."""
    from personal_kb.db.connection import create_connection

    db = await create_connection(":memory:", embedding_dim=64)

    now = datetime.now(UTC)

    entries = [
        # Created 2 days ago
        (
            "kb-00020",
            "proj",
            "Recent decision",
            "Made 2 days ago",
            "details",
            "decision",
            (now - timedelta(days=2)).isoformat(),
        ),
        # Created 10 days ago
        (
            "kb-00021",
            "proj",
            "Older decision",
            "Made 10 days ago",
            "details",
            "decision",
            (now - timedelta(days=10)).isoformat(),
        ),
        # Created 30 days ago
        (
            "kb-00022",
            "proj",
            "Old lesson",
            "Made 30 days ago",
            "details",
            "lesson_learned",
            (now - timedelta(days=30)).isoformat(),
        ),
        # Convention (always shown regardless of since)
        (
            "kb-00023",
            "proj",
            "Style guide",
            "Code style",
            "details",
            "pattern_convention",
            (now - timedelta(days=60)).isoformat(),
        ),
    ]

    for e in entries:
        await db.execute(
            "INSERT INTO knowledge_entries "
            "(id, project_ref, short_title, long_title, knowledge_details, "
            "entry_type, created_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            [*e, e[6]],  # updated_at = created_at
        )
    await db.commit()

    yield db
    await db.close()


class TestSinceFilter:
    async def test_since_7d_shows_recent_only(self, since_db) -> None:
        """since=7d shows only the 2-day-old decision, not older ones."""
        result = await build_project_context(since_db, "proj", since=timedelta(days=7))
        assert "kb-00020" in result  # 2 days ago
        assert "kb-00021" not in result  # 10 days ago
        assert "kb-00022" not in result  # 30 days ago
        # Convention always shown
        assert "kb-00023" in result

    async def test_since_2w_shows_recent_two(self, since_db) -> None:
        """since=14d shows both recent entries."""
        result = await build_project_context(since_db, "proj", since=timedelta(days=14))
        assert "kb-00020" in result
        assert "kb-00021" in result
        assert "kb-00022" not in result

    async def test_no_since_shows_all(self, since_db) -> None:
        """Without since, all decisions/lessons shown (up to limit)."""
        result = await build_project_context(since_db, "proj")
        assert "kb-00020" in result
        assert "kb-00021" in result
        assert "kb-00022" in result

    async def test_since_label_in_header(self, since_db) -> None:
        """Section header shows the time window."""
        result = await build_project_context(since_db, "proj", since=timedelta(days=7))
        assert "last 1w" in result

    async def test_since_weeks_label(self, since_db) -> None:
        result = await build_project_context(since_db, "proj", since=timedelta(weeks=2))
        assert "last 2w" in result


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
        result = await build_project_context(team_db, "shared-proj")
        assert "kb-00010" in result  # alpha
        assert "kb-00011" in result  # beta
        assert "kb-00012" in result  # global

    async def test_team_alpha_sees_own_and_global(self, team_db) -> None:
        """team-alpha sees alpha entries + global, not beta."""
        result = await build_project_context(team_db, "shared-proj", team="team-alpha")
        assert "kb-00010" in result  # alpha decision
        assert "kb-00012" in result  # global decision
        assert "kb-00013" in result  # alpha convention
        assert "kb-00015" in result  # alpha expiring
        assert "kb-00011" not in result  # beta decision
        assert "kb-00014" not in result  # beta convention

    async def test_team_beta_sees_own_and_global(self, team_db) -> None:
        """team-beta sees beta entries + global, not alpha."""
        result = await build_project_context(team_db, "shared-proj", team="team-beta")
        assert "kb-00011" in result  # beta decision
        assert "kb-00012" in result  # global decision
        assert "kb-00014" in result  # beta convention
        assert "kb-00010" not in result  # alpha decision
        assert "kb-00013" not in result  # alpha convention
        assert "kb-00015" not in result  # alpha expiring


# ---------------------------------------------------------------------------
# Graph expansion (2-hop via shared tags)
# ---------------------------------------------------------------------------


async def _insert_entry(db, entry_id, project_ref, short_title, entry_type, *, now):
    """Helper to insert an entry + its graph node."""
    await db.execute(
        "INSERT INTO knowledge_entries "
        "(id, project_ref, short_title, long_title, knowledge_details, "
        "entry_type, created_at, updated_at, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
        [
            entry_id,
            project_ref,
            short_title,
            short_title,
            "details",
            entry_type,
            now.isoformat(),
            now.isoformat(),
        ],
    )
    # graph_edges has FK to graph_nodes — entry must exist as a node
    await db.execute(
        "INSERT OR IGNORE INTO graph_nodes (node_id, node_type, created_at) VALUES (?, 'entry', ?)",
        [entry_id, now.isoformat()],
    )


async def _add_tag_edge(db, entry_id, tag_name, *, now):
    """Helper to create tag node + edge."""
    node_id = f"tag:{tag_name}"
    await db.execute(
        "INSERT OR IGNORE INTO graph_nodes (node_id, node_type, created_at) VALUES (?, 'tag', ?)",
        [node_id, now.isoformat()],
    )
    await db.execute(
        "INSERT OR IGNORE INTO graph_edges "
        "(source, target, edge_type, created_at) "
        "VALUES (?, ?, 'has_tag', ?)",
        [entry_id, node_id, now.isoformat()],
    )


@pytest.fixture()
async def graph_db():
    """DB with cross-project graph connections via shared tags.

    Project "alpha":
      kb-00030 decision  — tagged #api, #auth
      kb-00031 lesson    — tagged #api, #auth
      kb-00032 convention — tagged #api

    Project "beta":
      kb-00033 decision  — tagged #api, #auth  (should appear: shares 2+ tags with alpha)
      kb-00034 lesson    — tagged #api          (should appear: #api on 3 alpha entries)
      kb-00035 convention — tagged #api, #auth  (should NOT appear: wrong entry_type)

    Project "gamma":
      kb-00036 decision  — tagged #unrelated    (should NOT appear: no shared tags)
    """
    from personal_kb.db.connection import create_connection

    db = await create_connection(":memory:", embedding_dim=64)
    now = datetime.now(UTC)

    # Alpha project entries
    await _insert_entry(db, "kb-00030", "alpha", "Alpha API decision", "decision", now=now)
    await _insert_entry(db, "kb-00031", "alpha", "Alpha auth lesson", "lesson_learned", now=now)
    await _insert_entry(
        db,
        "kb-00032",
        "alpha",
        "Alpha API convention",
        "pattern_convention",
        now=now,
    )

    # Beta project entries
    await _insert_entry(db, "kb-00033", "beta", "Beta API+auth decision", "decision", now=now)
    await _insert_entry(db, "kb-00034", "beta", "Beta API-only lesson", "lesson_learned", now=now)
    await _insert_entry(db, "kb-00035", "beta", "Beta convention", "pattern_convention", now=now)

    # Gamma project entries
    await _insert_entry(db, "kb-00036", "gamma", "Gamma unrelated", "decision", now=now)

    # Tag edges — alpha entries
    await _add_tag_edge(db, "kb-00030", "api", now=now)
    await _add_tag_edge(db, "kb-00030", "auth", now=now)
    await _add_tag_edge(db, "kb-00031", "api", now=now)
    await _add_tag_edge(db, "kb-00031", "auth", now=now)
    await _add_tag_edge(db, "kb-00032", "api", now=now)

    # Tag edges — beta entries
    await _add_tag_edge(db, "kb-00033", "api", now=now)
    await _add_tag_edge(db, "kb-00033", "auth", now=now)
    await _add_tag_edge(db, "kb-00034", "api", now=now)
    await _add_tag_edge(db, "kb-00035", "api", now=now)
    await _add_tag_edge(db, "kb-00035", "auth", now=now)

    # Tag edges — gamma (no shared tags)
    await _add_tag_edge(db, "kb-00036", "unrelated", now=now)

    await db.commit()
    yield db
    await db.close()


class TestGraphExpansion:
    async def test_cross_project_decision_via_shared_tags(self, graph_db) -> None:
        """Beta decision with shared tags appears in alpha preflight."""
        result = await build_project_context(graph_db, "alpha")
        assert "Related (via graph):" in result
        assert "kb-00033" in result  # beta decision, shares #api + #auth

    async def test_excludes_wrong_entry_type(self, graph_db) -> None:
        """Conventions from other projects are excluded."""
        result = await build_project_context(graph_db, "alpha")
        assert "kb-00035" not in result  # convention, not decision/lesson

    async def test_excludes_unrelated_projects(self, graph_db) -> None:
        """Entries with no shared tags don't appear."""
        result = await build_project_context(graph_db, "alpha")
        assert "kb-00036" not in result  # gamma, no shared tags

    async def test_shows_via_tag_label(self, graph_db) -> None:
        """Related entries show which tag connected them."""
        result = await build_project_context(graph_db, "alpha")
        # Should show "(via #api)" or "(via #auth)"
        assert "(via #" in result

    async def test_no_graph_section_without_shared_tags(self, graph_db) -> None:
        """Gamma has no shared tags — no Related section."""
        result = await build_project_context(graph_db, "gamma")
        assert "Related (via graph):" not in result

    async def test_related_included_in_total_count(self, graph_db) -> None:
        """Total entry count includes graph-expanded entries."""
        result = await build_project_context(graph_db, "alpha")
        # alpha has: 2 decisions/lessons + 1 convention + graph entries
        # kb-00033 from beta via graph, kb-00034 via #api (if #api has 2+ alpha entries)
        lines = result.split("\n")
        header = lines[0]
        # Extract count from "alpha (N entries)"
        count = int(header.split("(")[1].split(" ")[0])
        assert count > 3  # at least the 3 alpha entries + some graph

    async def test_tag_needs_two_plus_project_entries(self) -> None:
        """A tag on only 1 project entry doesn't trigger expansion."""
        from personal_kb.db.connection import create_connection

        db = await create_connection(":memory:", embedding_dim=64)
        now = datetime.now(UTC)

        # Project with 1 entry tagged #rare
        await _insert_entry(db, "kb-00040", "solo", "Solo decision", "decision", now=now)
        await _add_tag_edge(db, "kb-00040", "rare", now=now)

        # Another project with entry tagged #rare
        await _insert_entry(db, "kb-00041", "other", "Other decision", "decision", now=now)
        await _add_tag_edge(db, "kb-00041", "rare", now=now)

        await db.commit()

        result = await build_project_context(db, "solo")
        # #rare only on 1 solo entry — no expansion
        assert "Related (via graph):" not in result
        assert "kb-00041" not in result

        await db.close()

    async def test_deactivated_entries_excluded(self) -> None:
        """Deactivated entries from other projects don't appear in graph."""
        from personal_kb.db.connection import create_connection

        db = await create_connection(":memory:", embedding_dim=64)
        now = datetime.now(UTC)

        await _insert_entry(db, "kb-00050", "proj-a", "A1", "decision", now=now)
        await _insert_entry(db, "kb-00051", "proj-a", "A2", "lesson_learned", now=now)
        await _add_tag_edge(db, "kb-00050", "shared", now=now)
        await _add_tag_edge(db, "kb-00051", "shared", now=now)

        # Deactivated entry in other project with shared tag
        await db.execute(
            "INSERT INTO knowledge_entries "
            "(id, project_ref, short_title, long_title, knowledge_details, "
            "entry_type, created_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            [
                "kb-00052",
                "proj-b",
                "Deactivated",
                "Deactivated",
                "details",
                "decision",
                now.isoformat(),
                now.isoformat(),
            ],
        )
        await db.execute(
            "INSERT OR IGNORE INTO graph_nodes (node_id, node_type, created_at) "
            "VALUES (?, 'entry', ?)",
            ["kb-00052", now.isoformat()],
        )
        await _add_tag_edge(db, "kb-00052", "shared", now=now)

        await db.commit()

        result = await build_project_context(db, "proj-a")
        assert "kb-00052" not in result

        await db.close()
