from __future__ import annotations

import fastapi

import qjob.api.crud as crud
import qjob.api.schemas as schemas

# --------------------------------------------------------------------------------------
# Router

router = fastapi.APIRouter(prefix="/resources", tags=["resources"])


# --------------------------------------------------------------------------------------
# GET /resources


@router.get(
    "",
    response_model=schemas.ResourceResponse,
    summary="Get resource availability",
)
def get_resources() -> schemas.ResourceResponse:
    """
    Return the current resource configuration and usage summary.

    Parameters
    ----------
    None

    Returns
    -------
    schemas.ResourceResponse
        Total and used resource counts.
    """

    info = crud.get_resources()

    return _info_to_response(info)


# --------------------------------------------------------------------------------------
# PUT /resources


@router.put(
    "",
    response_model=schemas.ResourceResponse,
    summary="Update resource limits (admin)",
)
def update_resources(body: schemas.ResourceUpdateRequest) -> schemas.ResourceResponse:
    """
    Update the available resource limits.

    Only the fields that are not None are updated.  This endpoint is
    intended for administrators only; access control should be enforced
    at the network or proxy layer in production.

    Parameters
    ----------
    body : schemas.ResourceUpdateRequest
        The new resource limits.  At least one field must be set.

    Returns
    -------
    schemas.ResourceResponse
        The updated resource configuration.

    Raises
    ------
    fastapi.HTTPException
        400 if all fields in *body* are None (caught from service layer).
    """

    try:
        info = crud.set_resources(
            total_cpus=body.total_cpus,
            total_gpus=body.total_gpus,
            total_mem_mb=body.total_mem_mb,
            max_walltime_sec=body.max_walltime_sec,
        )
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc))

    return _info_to_response(info)


# --------------------------------------------------------------------------------------
# Private helpers


def _info_to_response(info: crud.ResourceInfo) -> schemas.ResourceResponse:
    """
    Convert a ResourceInfo data class to a ResourceResponse schema.

    Parameters
    ----------
    info : crud.ResourceInfo
        The service-layer data object.

    Returns
    -------
    schemas.ResourceResponse
        The API response model.
    """

    return schemas.ResourceResponse(
        total_cpus=info.total_cpus,
        total_gpus=info.total_gpus,
        total_mem_mb=info.total_mem_mb,
        max_walltime_sec=info.max_walltime_sec,
        used_cpus=info.used_cpus,
        used_gpus=info.used_gpus,
        used_mem_mb=info.used_mem_mb,
    )
