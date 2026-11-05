from collections.abc import Iterator

from fastapi import Request
from sqlmodel import Session


def get_ctx(request: Request):
    return request.app.state.ctx


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.ctx.engine) as session:
        yield session
