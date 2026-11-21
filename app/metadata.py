import os
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from app.schemas import Citation, Work


def normalize_crossref(item: dict) -> Work:
    dates = item.get("published", {}).get("date-parts", [])
    year = dates[0][0] if dates and dates[0] else None
    titles = item.get("title") or []
    return Work(title=titles[0] if titles else None,
                authors=[" ".join(filter(None, [author.get("given"), author.get("family")]))
                         for author in item.get("author", [])],
                year=year, doi=item.get("DOI"), source="crossref",
                source_url=item.get("URL"))


def normalize_openalex(item: dict) -> Work:
    return Work(title=item.get("title"),
                authors=[entry.get("author", {}).get("display_name", "")
                         for entry in item.get("authorships", [])],
                year=item.get("publication_year"), doi=item.get("doi"),
                source="openalex", source_url=item.get("id"))


async def fetch_candidates(client: httpx.AsyncClient, citation: Citation) -> tuple[list[Work], bool]:
    works: list[Work] = []
    any_success = False
    contact = os.getenv("CROSSREF_CONTACT")
    headers = {"User-Agent": f"CheckMyCitation/0.1 (mailto:{contact})" if contact
               else "CheckMyCitation/0.1"}
    key = os.getenv("OPENALEX_API_KEY")
    openalex_params = {"api_key": key} if key else {}

    def add(item: dict, source: str) -> None:
        try:
            work = normalize_crossref(item) if source == "crossref" else normalize_openalex(item)
            if work.title or work.doi:
                works.append(work)
        except (AttributeError, TypeError, ValueError, ValidationError):
            pass

    async def request(url: str, *, params: dict | None = None, crossref: bool = False,
                      list_result: bool = False):
        nonlocal any_success
        try:
            response = await client.get(url, params=params, headers=headers if crossref else None)
            if response.status_code == 404:
                any_success = True
                return None
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                return None
            payload = data.get("message") if crossref else data
            if not isinstance(payload, dict):
                return None
            if list_result and not isinstance(payload.get("items" if crossref else "results"), list):
                return None
            if not list_result and not (payload.get("title") or payload.get("DOI" if crossref else "doi")):
                return None
            any_success = True
            return data
        except (httpx.HTTPError, ValueError):
            return None

    if citation.doi:
        crossref = await request(f"https://api.crossref.org/works/{quote(citation.doi, safe='/')}", crossref=True)
        openalex = await request(f"https://api.openalex.org/works/https://doi.org/{quote(citation.doi, safe='/')}", params=openalex_params)
        if crossref and isinstance(crossref.get("message"), dict):
            add(crossref["message"], "crossref")
        if openalex:
            add(openalex, "openalex")
        if works:
            return works, any_success

    query = citation.title or citation.raw_citation
    crossref = await request("https://api.crossref.org/works", params={"query.bibliographic": query, "rows": 5}, crossref=True, list_result=True)
    openalex = await request("https://api.openalex.org/works", params={"search": query, "per_page": 5, **openalex_params}, list_result=True)
    if crossref and isinstance(crossref.get("message"), dict):
        items = crossref["message"].get("items")
        if isinstance(items, list):
            for item in items[:5]:
                if isinstance(item, dict):
                    add(item, "crossref")
    if openalex and isinstance(openalex.get("results"), list):
        for item in openalex["results"][:5]:
            if isinstance(item, dict):
                add(item, "openalex")
    return works, any_success
