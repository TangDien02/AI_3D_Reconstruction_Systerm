from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest


BLOGS_ENDPOINT = "/api/blogs"


def build_blog_payload(**overrides: object) -> dict[str, object]:
    unique_id = uuid4().hex[:10]
    payload: dict[str, object] = {
        "title": f"Automation blog {unique_id}",
        "content": f"This blog was created by pytest automation at {datetime.now(UTC).isoformat()}.",
    }
    payload.update(overrides)
    return payload


def assert_blog_schema(blog: dict[str, object]) -> None:
    # TODO: Adjust required fields if Postman response uses different names,
    # for example "_id" instead of "id", "body" instead of "content".
    required_fields = ["id", "title", "content"]
    for field in required_fields:
        assert field in blog, f"Missing required field: {field}. Response: {blog}"

    assert blog["id"] not in (None, "")
    assert isinstance(blog["title"], str)
    assert isinstance(blog["content"], str)

    # TODO: Enable these as strict asserts if API always returns timestamps.
    if "created_at" in blog:
        assert blog["created_at"] not in (None, "")
    if "updated_at" in blog:
        assert blog["updated_at"] not in (None, "")


def extract_blog_id(blog: dict[str, object]) -> str | int:
    # TODO: Change to "_id" if actual API uses Mongo-style identifiers.
    assert "id" in blog, f"Cannot find blog id in response: {blog}"
    blog_id = blog["id"]
    assert isinstance(blog_id, (str, int))
    assert blog_id != ""
    return blog_id


def parse_json_object(response: httpx.Response) -> dict[str, object]:
    data = response.json()
    assert isinstance(data, dict), f"Expected JSON object, got: {data}"
    return data


def parse_blog_from_response(response: httpx.Response) -> dict[str, object]:
    data = parse_json_object(response)

    # Supports common response shapes:
    # 1. {"id": 1, "title": "...", "content": "..."}
    # 2. {"data": {"id": 1, "title": "...", "content": "..."}}
    # TODO: Adjust this if Postman shows another wrapper, e.g. {"blog": {...}}.
    if "data" in data and isinstance(data["data"], dict):
        return data["data"]
    return data


@pytest.mark.blogs
def test_get_blogs_list(api_client: httpx.Client) -> None:
    response = api_client.get(BLOGS_ENDPOINT)

    assert response.status_code == 200
    data = response.json()

    # Supports common response shapes:
    # 1. [{"id": 1, ...}]
    # 2. {"data": [{"id": 1, ...}]}
    # TODO: Adjust if API returns pagination shape like {"items": [...], "total": 10}.
    blogs = data.get("data") if isinstance(data, dict) else data
    assert isinstance(blogs, list), f"Expected blogs list, got: {data}"

    if blogs:
        assert isinstance(blogs[0], dict)
        assert_blog_schema(blogs[0])


@pytest.mark.blogs
def test_create_blog(api_client: httpx.Client, created_blog_ids: list[str | int]) -> None:
    payload = build_blog_payload()

    response = api_client.post(BLOGS_ENDPOINT, json=payload)

    assert response.status_code in (200, 201)
    blog = parse_blog_from_response(response)
    assert_blog_schema(blog)
    assert blog["title"] == payload["title"]
    assert blog["content"] == payload["content"]

    created_blog_ids.append(extract_blog_id(blog))


@pytest.mark.blogs
def test_get_created_blog_detail(api_client: httpx.Client, created_blog_ids: list[str | int]) -> None:
    payload = build_blog_payload()
    create_response = api_client.post(BLOGS_ENDPOINT, json=payload)
    assert create_response.status_code in (200, 201)

    created_blog = parse_blog_from_response(create_response)
    blog_id = extract_blog_id(created_blog)
    created_blog_ids.append(blog_id)

    response = api_client.get(f"{BLOGS_ENDPOINT}/{blog_id}")

    assert response.status_code == 200
    blog = parse_blog_from_response(response)
    assert_blog_schema(blog)
    assert extract_blog_id(blog) == blog_id
    assert blog["title"] == payload["title"]
    assert blog["content"] == payload["content"]


@pytest.mark.blogs
def test_update_blog(api_client: httpx.Client, created_blog_ids: list[str | int]) -> None:
    payload = build_blog_payload()
    create_response = api_client.post(BLOGS_ENDPOINT, json=payload)
    assert create_response.status_code in (200, 201)

    created_blog = parse_blog_from_response(create_response)
    blog_id = extract_blog_id(created_blog)
    created_blog_ids.append(blog_id)

    update_payload = build_blog_payload(
        title=f"Updated {payload['title']}",
        content="Updated content from pytest automation.",
    )

    # TODO: Use PATCH instead of PUT if Postman confirms partial update is PATCH.
    response = api_client.put(f"{BLOGS_ENDPOINT}/{blog_id}", json=update_payload)

    assert response.status_code in (200, 202)
    updated_blog = parse_blog_from_response(response)
    assert_blog_schema(updated_blog)
    assert extract_blog_id(updated_blog) == blog_id
    assert updated_blog["title"] == update_payload["title"]
    assert updated_blog["content"] == update_payload["content"]

    detail_response = api_client.get(f"{BLOGS_ENDPOINT}/{blog_id}")
    assert detail_response.status_code == 200
    detail_blog = parse_blog_from_response(detail_response)
    assert detail_blog["title"] == update_payload["title"]
    assert detail_blog["content"] == update_payload["content"]


@pytest.mark.blogs
def test_delete_blog_and_verify_not_found(api_client: httpx.Client, created_blog_ids: list[str | int]) -> None:
    payload = build_blog_payload()
    create_response = api_client.post(BLOGS_ENDPOINT, json=payload)
    assert create_response.status_code in (200, 201)

    created_blog = parse_blog_from_response(create_response)
    blog_id = extract_blog_id(created_blog)
    created_blog_ids.append(blog_id)

    delete_response = api_client.delete(f"{BLOGS_ENDPOINT}/{blog_id}")

    assert delete_response.status_code in (200, 202, 204)
    created_blog_ids.remove(blog_id)

    get_response = api_client.get(f"{BLOGS_ENDPOINT}/{blog_id}")

    # TODO: If API returns 200 with {"deleted": true} or {"data": null},
    # replace this assert with the behavior observed in Postman.
    assert get_response.status_code == 404
