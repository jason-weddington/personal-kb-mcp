"""Settings routes: GET/PUT /api/settings (app-config key-value store).

Exposes the 'team' setting from the app_config table.  GET is available to
any authenticated user; PUT is admin-only and performs a full replace (an
omitted field clears the setting, same as an explicit null).
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from kb_service.attribution import _normalize, delete_setting, get_setting, set_setting
from kb_service.auth import get_current_user, require_admin
from kb_service.models import SettingsResponse, UpdateSettingsRequest, User

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(
    user: Annotated[User, Depends(get_current_user)],
) -> SettingsResponse:
    """Return current settings (any authenticated user).

    Returns the normalised 'team' value: absent or blank rows are returned as
    null, never as an empty string.
    """
    return SettingsResponse(team=_normalize(await get_setting("team")))


@router.put("", response_model=SettingsResponse)
async def put_settings(
    body: UpdateSettingsRequest,
    user: Annotated[User, Depends(require_admin)],
) -> SettingsResponse:
    """Replace settings (admin only).

    Full-replace semantics: an omitted ``team`` field is treated identically to
    ``team: null`` and clears the stored value.  Whitespace-only values are also
    treated as a clear.  The response reflects the post-write state.
    """
    if body.team is None or body.team.strip() == "":
        await delete_setting("team")
    else:
        await set_setting("team", body.team.strip(), updated_by=user.id)
    return SettingsResponse(team=_normalize(await get_setting("team")))
