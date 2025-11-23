import typer
from loguru import logger
from pathlib import Path


def work_dir_callback(ctx: typer.Context, param: typer.CallbackParam, value: str):
    if ctx.resilient_parsing:
        return
    logger.debug(f"Validating param: {param.name}")
    if param.name == "work_dir":
        work_dir = Path(value)
        logger.debug(f"Ensuring work_dir exists at: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
    return value
