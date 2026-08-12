"""Archive a tagged release to Zenodo without the GitHub integration.

Zenodo's GitHub webhook archives a repository automatically when a release is
published, which is the usual way to do this. It is also a single point of
failure: the OAuth token Zenodo stores for GitHub can expire, an organisation
can restrict third-party applications, and the failure surfaces as a 401 or a
400 with nothing archived. This script does the same job over the deposit API,
where the only credential involved is your own token.

    export ZENODO_PAT=...                       # zenodo.org > applications
    uv run python scripts/zenodo_release.py --tag v0.2.0        # make a draft
    uv run python scripts/zenodo_release.py --tag v0.2.0 --publish

It creates a **new version** of an existing record, so the concept DOI keeps
resolving to the latest and every earlier version DOI stays valid. Metadata
comes from `.zenodo.json`, with the version taken from the tag.

**It does not publish unless asked.** Publishing mints a DOI and cannot be
undone; the default stops at a draft and prints its URL so you can look at it
first.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ZENODO = "https://zenodo.org/api"

#: The record every new version descends from. Its concept DOI,
#: 10.5281/zenodo.3759707, is the one that always resolves to the latest.
DEFAULT_PARENT = "3759708"

#: The repository GitHub serves tag archives from.
DEFAULT_REPO = (
    "NeuroLang/Microstructural-organization-of-human-insula-"
    "linkded-circutry-predicts-congitive-abilities"
)


def api(method: str, url: str, token: str, data=None, content_type=None) -> dict:
    """One Zenodo API call, returning the decoded body."""
    request = urllib.request.Request(url, method=method, data=data)
    request.add_header("Authorization", f"Bearer {token}")
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
    except urllib.error.HTTPError as error:  # surface Zenodo's own message
        detail = error.read().decode("utf8", "replace")[:500]
        raise SystemExit(f"{method} {url}\n  HTTP {error.code}: {detail}") from None
    return json.loads(body) if body else {}


def fetch_tarball(repo: str, tag: str) -> bytes:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/tags/{tag}"
    with urllib.request.urlopen(url) as response:
        return response.read()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", required=True, help="git tag to archive, e.g. v0.2.0")
    parser.add_argument("--parent", default=DEFAULT_PARENT, help="record to version")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--publish", action="store_true",
        help="mint the DOI. Without this the script stops at a draft.",
    )
    parser.add_argument("--discard", action="store_true", help="delete the draft again")
    args = parser.parse_args(argv)

    token = os.environ.get("ZENODO_PAT")
    if not token:
        raise SystemExit("ZENODO_PAT is not set.")

    metadata = json.loads(Path(".zenodo.json").read_text())
    metadata["version"] = args.tag
    # The GitHub integration records the source tree it archived. Nothing else
    # ties the record to the code it came from, so add it here too.
    tree = f"https://github.com/{args.repo}/tree/{args.tag}"
    related = list(metadata.get("related_identifiers") or [])
    if not any(r.get("identifier") == tree for r in related):
        related.append(
            {"identifier": tree, "relation": "isSupplementTo", "scheme": "url"}
        )
    metadata["related_identifiers"] = related
    print(f"tag {args.tag}: fetching the archive from GitHub")
    archive = fetch_tarball(args.repo, args.tag)
    print(f"  {len(archive)} bytes")

    draft = api(
        "POST", f"{ZENODO}/deposit/depositions/{args.parent}/actions/newversion", token
    )
    new_id = draft["links"]["latest_draft"].rstrip("/").split("/")[-1]
    draft = api("GET", f"{ZENODO}/deposit/depositions/{new_id}", token)
    print(f"  draft {new_id} under concept {draft['conceptrecid']}")

    # A new version inherits the previous version's files. They are the previous
    # release, not this one, so they go.
    for stale in draft.get("files", []):
        api(
            "DELETE",
            f"{ZENODO}/deposit/depositions/{new_id}/files/{stale['id']}",
            token,
        )
        print(f"  dropped inherited {stale['filename']}")

    # Name the archive after the distribution, not the checkout directory,
    # which is whatever the person who cloned it chose.
    project = re.search(r'^name = "([^"]+)"', Path("pyproject.toml").read_text(), re.M)
    name = f"{project.group(1)}-{args.tag}.tar.gz"
    uploaded = api(
        "PUT", f"{draft['links']['bucket']}/{name}", token,
        data=archive, content_type="application/octet-stream",
    )
    print(f"  uploaded {uploaded['key']} ({uploaded['checksum']})")

    api(
        "PUT", f"{ZENODO}/deposit/depositions/{new_id}", token,
        data=json.dumps({"metadata": metadata}).encode(),
        content_type="application/json",
    )
    print("  metadata applied from .zenodo.json")

    if args.discard:
        api("DELETE", f"{ZENODO}/deposit/depositions/{new_id}", token)
        print("  draft discarded")
        return
    if not args.publish:
        print(f"\nDraft ready, nothing minted: https://zenodo.org/deposit/{new_id}")
        print("Re-run with --publish to mint the DOI.")
        return

    published = api(
        "POST", f"{ZENODO}/deposit/depositions/{new_id}/actions/publish", token
    )
    print(f"\npublished {published['doi']}  (concept {published['conceptdoi']})")
    print(f"  {published['links']['record_html']}")


if __name__ == "__main__":
    sys.exit(main())
