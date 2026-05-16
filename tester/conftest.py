from __future__ import annotations

import os
from collections.abc import Generator

import httpx
import pytest
from dotenv import load_dotenv


load_dotenv()


DEFAULT_TIMEOUT = 15.0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "blogs: API tests for blogs resource")


@pytest.fixture(scope="session")
def base_url() -> str:
    value = os.getenv("BASE_URL", "").strip().rstrip("/")
    if not value:
        pytest.fail("Missing BASE_URL. Copy .env.example to .env and set BASE_URL.")
    return value


@pytest.fixture(scope="session")
def api_token() -> str | None:
    value = os.getenv("API_TOKEN", "").strip()
    return value or None


@pytest.fixture(scope="session")
def auth_headers(api_token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_token:
        # TODO: Adjust auth scheme if Postman uses a different one:
        # examples: "Token <token>", "ApiKey <token>", or custom "x-api-key".
        headers["Authorization"] = f"Bearer {api_token}"
    return headers


@pytest.fixture(scope="session")
def api_client(base_url: str, auth_headers: dict[str, str]) -> Generator[httpx.Client, None, None]:
    with httpx.Client(
        base_url=base_url,
        headers=auth_headers,
        timeout=httpx.Timeout(DEFAULT_TIMEOUT),
        follow_redirects=True,
    ) as client:
        yield client


@pytest.fixture
def created_blog_ids(api_client: httpx.Client) -> Generator[list[str | int], None, None]:
    ids: list[str | int] = []
    yield ids

    for blog_id in ids:
        try:
            response = api_client.delete(f"/api/blogs/{blog_id}")
            # Cleanup should not fail the test if the test already deleted the blog.
            if response.status_code not in (200, 202, 204, 404):
                print(f"Cleanup warning: DELETE /api/blogs/{blog_id} returned {response.status_code}")
        except httpx.HTTPError as exc:
            print(f"Cleanup warning: could not delete blog {blog_id}: {exc}")
