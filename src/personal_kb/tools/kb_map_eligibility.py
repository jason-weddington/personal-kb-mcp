"""kb_map_eligibility MCP tools — map-eligibility review + human override."""

import logging
import re
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

logger = logging.getLogger(__name__)

# Same literal as the owner item's kb-core constant, so one `grep
# map-eligibility` spans both ~/.local/share/personal_kb/log.txt and
# journalctl -u kb-service.
MAP_ELIGIBILITY_MARKER = "map-eligibility"

_PAYLOAD_ERROR = (
    "Error: KB service returned an unexpected map-eligibility payload — missing field {key}. "
    "The service may be running a version older than this client."
)


def _eligibility_description(prefix: str) -> str:
    """Build the kb_map_eligibility description with correct tool cross-references."""
    return (
        "Review per-project map eligibility across the KB — the full table a human "
        "or agent reads before deciding which projects deserve a mental map.\n\n"
        "Each row carries an evidence block and an effective verdict. Evidence fields:\n"
        "- project_ref: the project the row describes.\n"
        "- mappable: total active, non-expired entries counted as mappable for this "
        "project.\n"
        "- ingested: how many of those arrived via file/URL ingestion rather than "
        "hand-authoring.\n"
        "- hand_authored: mappable minus ingested — entries a human actually wrote.\n"
        "- maps: the project's existing mental_map entry count, so `eligible AND "
        "maps == 0` is the actual review target.\n"
        "- top_prefix: the most common leading token across the project's entry titles.\n"
        "- top_prefix_share: that token's share of titles, as an unrounded float.\n"
        "- is_ingest_corpus when ingested == mappable (reporting only, not a verdict "
        "term).\n"
        "- is_too_thin when hand_authored < 5 — too little human-authored knowledge "
        "to be worth a map.\n"
        "- is_journal when mappable >= 20 and top_prefix_share >= 0.60 — i.e. one "
        "prefix holding 60% or more of the titles, the shape of a dated session "
        "journal rather than a topical partition.\n"
        "- computed_eligible: the engine's verdict from the evidence alone.\n\n"
        "An eligible project's INGESTED entries count as mappable, because "
        "corpus-ness is a property of the PROJECT, not of an entry's provenance.\n"
        "decided_by == 'override' means a human verdict is in force, beating the "
        "computed verdict until cleared. orphaned means the row's project_ref has "
        "no mappable entries — stale, renamed or misspelled.\n\n"
        f"To change a verdict, use {prefix}map_eligibility_override."
    )


def _override_description(prefix: str) -> str:
    """Build the kb_map_eligibility_override description with tool cross-references."""
    return (
        "Set or clear a human map-eligibility override for one project_ref.\n\n"
        "reason is mandatory and permanent — it is stored as the audit trail for "
        "this human verdict. An override beats the computed verdict in BOTH "
        "directions until cleared. clear=True removes the override so the project "
        "reverts to the computed verdict.\n\n"
        "This is the human/agent-review path only: the nightly map-maintenance "
        "loop READS this table and never writes it.\n\n"
        "Worked example: harness-design passes every computed gate (71 mappable, "
        "0 ingested) yet is a dated session journal — a reading order with no "
        "topical partition — so it is the case for eligible=False. threat-intel "
        "sits exactly at the hand_authored = 5 floor, so it is the case for a "
        "deliberate judgement either way.\n\n"
        f"Run {prefix}map_eligibility to see the table this tool writes into."
    )


def _render_verdicts(rows: list[dict[str, Any]], prefix: str) -> str:
    """Render the map-eligibility review table.

    The header's counts are deliberately the same tuple, in the same order, as
    the service-side INFO summary the kb-core item's AC11 requires, so
    ``journalctl -u kb-service | grep map-eligibility`` can be diffed against
    this header to confirm the payload survived the wire.
    """
    try:
        if not rows:
            return (
                "No project_refs returned — the KB has no entries with a project_ref, "
                "or the service returned an empty list."
            )

        total = len(rows)
        n_elig = sum(1 for r in rows if r["effective_eligible"])
        n_ec = sum(1 for r in rows if r["effective_eligible"] and r["decided_by"] == "computed")
        n_eo = sum(1 for r in rows if r["effective_eligible"] and r["decided_by"] == "override")
        n_inel = total - n_elig
        n_eu = sum(1 for r in rows if r["effective_eligible"] and r["evidence"]["maps"] == 0)
        n_tt = sum(1 for r in rows if r["evidence"]["is_too_thin"])
        n_ic = sum(1 for r in rows if r["evidence"]["is_ingest_corpus"])
        n_j = sum(1 for r in rows if r["evidence"]["is_journal"])
        n_ov = sum(1 for r in rows if r["override"] is not None)

        header = (
            f"Map eligibility — {total} projects | eligible {n_elig} "
            f"(computed {n_ec}, override {n_eo}) | ineligible {n_inel} "
            f"| eligible & unmapped {n_eu} | evidence flags: too_thin {n_tt}, "
            f"ingest_corpus {n_ic}, journal {n_j} | override rows {n_ov}"
        )

        lines = [header, ""]
        n_missing = 0
        for row in rows:
            ev = row["evidence"]
            project_ref = ev["project_ref"]
            effective = row["effective_eligible"]
            decided_by = row["decided_by"]
            mappable = ev["mappable"]
            hand_authored = ev["hand_authored"]
            ingested = ev["ingested"]
            maps = ev["maps"]
            top_prefix = ev["top_prefix"]
            share = ev["top_prefix_share"]

            flags: list[str] = []
            if ev["is_too_thin"]:
                flags.append("too_thin")
            if ev["is_ingest_corpus"]:
                flags.append("ingest_corpus")
            if ev["is_journal"]:
                flags.append("journal")
            flags_text = ", ".join(flags) if flags else "none"

            line = (
                f"{project_ref} | {'ELIGIBLE' if effective else 'INELIGIBLE'} "
                f"({decided_by}) | mappable {mappable} = hand {hand_authored} "
                f"+ ingested {ingested} | maps {maps} | top '{top_prefix}' "
                f"{share:.1%} | flags: {flags_text}"
            )
            if row["orphaned"]:
                line += (
                    " | ORPHANED (0 mappable entries — stale, renamed or misspelled project_ref)"
                )
            lines.append(line)

            override = row["override"]
            if override is not None:
                set_by = override["set_by"]
                if not set_by:
                    n_missing += 1
                lines.append(
                    f"  override: {'eligible' if override['eligible'] else 'ineligible'} "
                    f"— {override['reason']} | set_by "
                    f"{set_by or 'MISSING — service recorded no identity'} "
                    f"| set_at {override['set_at']} | computed said "
                    f"{'eligible' if ev['computed_eligible'] else 'ineligible'}"
                )

        lines.append("")
        lines.append(
            "Thresholds: too_thin when hand_authored < 5; journal when mappable >= 20 "
            "and top prefix share >= 60%. To change a verdict: "
            f"{prefix}map_eligibility_override(project_ref=..., eligible=..., "
            'reason="...").'
        )

        if n_missing:
            logger.warning(
                "%s: %d of %d override rows have no set_by — the service is not "
                "threading the authenticated principal into kb-core",
                MAP_ELIGIBILITY_MARKER,
                n_missing,
                n_ov,
            )
        return "\n".join(lines)
    except KeyError as exc:
        logger.warning(
            "%s: verdict row missing field %s — MCP client and kb-service wire "
            "contracts have drifted",
            MAP_ELIGIBILITY_MARKER,
            exc,
        )
        return _PAYLOAD_ERROR.format(key=exc)


def register_kb_map_eligibility(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_map_eligibility tool with the MCP server."""

    @mcp.tool(name=f"{prefix}map_eligibility", description=_eligibility_description(prefix))
    async def kb_map_eligibility(
        project_ref: Annotated[
            str | None,
            Field(
                description=(
                    "Optional — narrow the report to one project_ref. "
                    "Omit for the full review table."
                )
            ),
        ] = None,
        ctx: Context | None = None,
    ) -> str:
        """Render the map-eligibility review table for the whole KB."""
        from personal_kb.tools._lifespan import backend_from_lifespan

        if ctx is None:
            raise RuntimeError("Context not injected")

        backend = backend_from_lifespan(ctx.lifespan_context)

        try:
            all_rows = await backend.map_eligibility()
            rows = (
                [r for r in all_rows if r["evidence"]["project_ref"] == project_ref]
                if project_ref is not None
                else all_rows
            )
            if all_rows and not rows:
                return (
                    f"No map-eligibility row for project_ref '{project_ref}'. "
                    f"Use {prefix}list_projects to see valid project_refs."
                )
            return _render_verdicts(rows, prefix)
        except KeyError as exc:
            logger.warning(
                "%s: verdict row missing field %s — MCP client and kb-service wire "
                "contracts have drifted",
                MAP_ELIGIBILITY_MARKER,
                exc,
            )
            return _PAYLOAD_ERROR.format(key=exc)
        except Exception as exc:
            from personal_kb.backend.http import BackendHttpError, _map_error

            if isinstance(exc, BackendHttpError):
                if exc.status == 404:
                    logger.warning(
                        "%s: this kb-service has no map-eligibility endpoint (404) — "
                        "client is newer than the deployed service",
                        MAP_ELIGIBILITY_MARKER,
                    )
                    return (
                        "Error: this KB service has no map-eligibility endpoint (404) — "
                        "it is probably running a version older than this MCP client. "
                        "Check the deployed kb-service version."
                    )
                return _map_error(exc, "")
            return f"Error: {exc}"


def register_kb_map_eligibility_override(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_map_eligibility_override tool with the MCP server."""

    @mcp.tool(
        name=f"{prefix}map_eligibility_override",
        description=_override_description(prefix),
    )
    async def kb_map_eligibility_override(
        project_ref: Annotated[
            str,
            Field(description="The project_ref whose verdict you are overriding."),
        ],
        eligible: Annotated[
            bool | None,
            Field(
                description=(
                    "Force the verdict: True = map this project, False = do not. "
                    "Required unless clear=True."
                )
            ),
        ] = None,
        reason: Annotated[
            str | None,
            Field(
                description=(
                    "Why. Stored permanently as the audit trail for this human "
                    "verdict. Required unless clear=True. The service caps this "
                    "at 2000 characters."
                )
            ),
        ] = None,
        clear: Annotated[
            bool,
            Field(
                description=(
                    "True removes the override so the project reverts to the computed verdict."
                )
            ),
        ] = False,
        ctx: Context | None = None,
    ) -> str:
        """Set or clear a human map-eligibility override for one project_ref."""
        from personal_kb.tools._lifespan import backend_from_lifespan

        # Validation before any HTTP — a malformed call costs no round trip.
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", project_ref):
            return (
                "Error: project_ref must be non-empty and contain only letters, "
                "digits, '_', '.' or '-'."
            )

        if ctx is None:
            raise RuntimeError("Context not injected")

        backend = backend_from_lifespan(ctx.lifespan_context)

        try:
            if clear:
                if eligible is not None or reason is not None:
                    return "Error: clear=True takes only project_ref — omit eligible and reason."
                cleared = await backend.clear_map_eligibility_override(project_ref)
                if cleared:
                    return (
                        f"Override cleared — {project_ref} reverts to the computed "
                        f"verdict. Run {prefix}map_eligibility to see it."
                    )
                return f"No override existed for {project_ref} — nothing to clear."

            if eligible is None:
                return "Error: eligible is required (True or False) unless clear=True."
            if reason is None or not reason.strip():
                return (
                    "Error: reason is required — the override is a human verdict and "
                    "the reason is its audit trail."
                )
            if len(reason) > 2000:
                return (
                    "Error: reason exceeds the service's 2000-character cap — shorten it and retry."
                )

            result = await backend.set_map_eligibility_override(
                project_ref, eligible=eligible, reason=reason
            )
            verdict = result["verdict"]
            if verdict is None:
                logger.warning(
                    "%s: set for %s returned no verdict — the service logged an invariant breach",
                    MAP_ELIGIBILITY_MARKER,
                    project_ref,
                )
                return (
                    f"Override stored for {project_ref}, but the service returned no "
                    "resolved verdict — it logs this as an invariant breach. "
                    f"Run {prefix}map_eligibility to confirm the override is in force."
                )

            effective = verdict["effective_eligible"]
            decided_by = verdict["decided_by"]
            orphaned = verdict["orphaned"]
            override = verdict.get("override") or {}
            set_by = override.get("set_by")
            set_at = override.get("set_at", "unknown")
            # Render the STORED reason, not the caller's: the service strips
            # whitespace before persisting, so echoing the request text would
            # misreport what the audit trail holds.
            stored_reason = override.get("reason", reason)

            msg = (
                f"Override set — {project_ref} is now "
                f"{'ELIGIBLE' if effective else 'INELIGIBLE'} ({decided_by}), beating "
                f"the computed verdict until cleared. reason: {stored_reason} | "
                f"set_by {set_by or 'MISSING'} | set_at {set_at}"
            )
            if not set_by:
                msg += (
                    " WARNING: the service recorded no set_by for this override — "
                    "its audit trail has no attribution. The route is not passing "
                    "the authenticated principal to kb-core."
                )
                logger.warning(
                    "%s: override set for %s returned no set_by — audit trail has no attribution",
                    MAP_ELIGIBILITY_MARKER,
                    project_ref,
                )
            if orphaned:
                msg += (
                    f" WARNING: {project_ref} has no mappable entries — the override "
                    "is stored but will never affect anything. Check the spelling "
                    f"with {prefix}list_projects."
                )
            return msg
        except KeyError as exc:
            logger.warning(
                "%s: verdict row missing field %s — MCP client and kb-service wire "
                "contracts have drifted",
                MAP_ELIGIBILITY_MARKER,
                exc,
            )
            return _PAYLOAD_ERROR.format(key=exc)
        except Exception as exc:
            from personal_kb.backend.http import BackendHttpError, _map_error

            if isinstance(exc, BackendHttpError):
                if exc.status == 404:
                    logger.warning(
                        "%s: this kb-service has no map-eligibility endpoint (404) — "
                        "client is newer than the deployed service",
                        MAP_ELIGIBILITY_MARKER,
                    )
                    return (
                        "Error: this KB service has no map-eligibility endpoint (404) — "
                        "it is probably running a version older than this MCP client. "
                        "Check the deployed kb-service version."
                    )
                return _map_error(exc, "")
            return f"Error: {exc}"
