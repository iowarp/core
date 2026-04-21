
#!/usr/bin/env python3
"""
get_oauth_token.py - Get Globus OAuth2 tokens programmatically

This demonstrates how to get Globus access tokens using the OAuth2 flow
for REST API access to download data.

For production use, you should:
1. Register a Native App or Confidential Client at https://app.globus.org/settings/developers
2. Use the client ID and implement proper OAuth2 flow

This script uses the Globus SDK's Native App flow.
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import globus_sdk

# Transfer API base URL for fetching public endpoint metadata
TRANSFER_API_BASE = "https://transfer.api.globus.org/v0.10"


def get_endpoint_metadata(collection_id):
    """
    Fetch endpoint/collection document from the Transfer API (no auth).
    Works for public endpoints; returns None if the request fails (e.g. private
    endpoint or network error).

    Args:
        collection_id: The Globus collection/endpoint ID.

    Returns:
        Dict with at least entity_type and high_assurance, or None on failure.
    """
    url = f"{TRANSFER_API_BASE}/endpoint/{collection_id}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def scopes_for_collection(collection_id, endpoint_doc=None, force_data_access=False):
    """
    Build the list of OAuth scopes needed for Transfer + HTTPS access to a
    collection, based on collection type so that data_access is only requested
    when required (non-High-Assurance mapped collections).

    Args:
        collection_id: The Globus collection/endpoint ID.
        endpoint_doc: Optional endpoint document from Transfer API. If None,
            endpoint metadata is fetched (for public endpoints).
        force_data_access: If True, always include data_access scope (use when
            you know the collection is a non-HA mapped collection and you
            got a consent error without it).

    Returns:
        Tuple of (list of scope strings, need_data_access: bool).
    """
    transfer_scope = "urn:globus:auth:scope:transfer.api.globus.org:all"
    https_scope = f"https://auth.globus.org/scopes/{collection_id}/https"
    data_access_scope = f"https://auth.globus.org/scopes/{collection_id}/data_access"

    if endpoint_doc is None:
        endpoint_doc = get_endpoint_metadata(collection_id)

    need_data_access = False
    if endpoint_doc:
        entity_type = endpoint_doc.get("entity_type") or ""
        high_assurance = endpoint_doc.get("high_assurance") is True
        # data_access is only valid for non-HA GCSv5 mapped collections. Guest
        # collections use the guest ID; data_access is on the mapped collection,
        # so requesting it for a guest ID yields "unknown scope".
        if entity_type == "GCSv5_mapped_collection" and not high_assurance:
            need_data_access = True
    # When metadata is unavailable, omit data_access to avoid "unknown scope"
    # for guest collections; if the collection is mapped and needs it, the user
    # will see a consent error when using the token and can retry with
    # --with-data-access.
    if force_data_access:
        need_data_access = True

    scopes = [transfer_scope, https_scope]
    if need_data_access:
        scopes.append(data_access_scope)
    return (scopes, need_data_access)


def get_tokens_for_collection(collection_id, client_id, force_data_access=False):
    """
    Get OAuth2 tokens for Transfer API and HTTPS access to a collection.

    Args:
        collection_id: The Globus collection/endpoint ID
        client_id: Native app client ID from Globus.
        force_data_access: If True, always request data_access scope.

    Returns:
        Dictionary with access tokens
    """
    scopes, need_data_access = scopes_for_collection(
        collection_id, force_data_access=force_data_access
    )

    print("=== Globus OAuth2 Token Generation ===")
    print("")
    print(f"Collection ID: {collection_id}")
    print("Required scopes (chosen from collection type):")
    for scope in scopes:
        print(f"  - {scope}")
    if need_data_access:
        print("  (data_access included)")
    else:
        print("  (data_access omitted; if you see a consent error, try --with-data-access)")
    print("")

    # Create a Native App client
    client = globus_sdk.NativeAppAuthClient(client_id)

    # Start the OAuth2 flow
    client.oauth2_start_flow(requested_scopes=scopes)

    # Get the authorization URL
    authorize_url = client.oauth2_get_authorize_url()

    # Save PKCE verifier so --auth-code can complete the exchange later
    _pkce_state = {
        "verifier": client.current_oauth2_flow_manager.verifier,
        "scopes": scopes,
        "collection_id": collection_id,
        "client_id": client_id,
    }
    with open("/tmp/globus_pkce_state.json", "w") as _f:
        json.dump(_pkce_state, _f)

    print("Please visit this URL to authorize the application:")
    print("")
    print(authorize_url)
    print("")
    print("Then re-run with: --auth-code <code>")
    print("")

    # Wait for the user to authorize and paste the code
    auth_code = input("Paste the authorization code here: ").strip()

    # Exchange the code for tokens
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)

    # Get tokens for each resource server
    transfer_tokens = token_response.by_resource_server['transfer.api.globus.org']
    collection_tokens = token_response.by_resource_server.get(collection_id)

    # The HTTPS scope is a dependent scope, so it might be in the transfer tokens
    # or we need to get it separately

    print("")
    print("=== Tokens Retrieved ===")
    print("")
    print("Transfer API Access Token:")
    print(transfer_tokens['access_token'])
    print("")

    if collection_tokens:
        print(f"Collection Access Token ({collection_id}):")
        print(collection_tokens['access_token'])
        print("")

    # Check if we have HTTPS scope tokens
    # For dependent scopes, they may be included in the transfer token response
    print("You can now use tokens with:")
    print(f"  export GLOBUS_ACCESS_TOKEN='{transfer_tokens['access_token']}'")
    if collection_tokens:
        print(
            "  export GLOBUS_HTTPS_ACCESS_TOKEN="
            f"'{collection_tokens['access_token']}'"
        )
    print("")

    txt_path = "/tmp/globus_tokens.txt"
    json_path = "/tmp/globus_tokens.json"
    sh_path = "/tmp/globus_tokens.sh"

    # Save tokens to TXT format (KEY=VALUE)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"GLOBUS_ACCESS_TOKEN={transfer_tokens['access_token']}\n")
        if collection_tokens:
            f.write(f"GLOBUS_HTTPS_ACCESS_TOKEN={collection_tokens['access_token']}\n")

    # Save full token response JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(token_response.data, f, indent=2)

    # Save source-able shell script
    with open(sh_path, 'w', encoding='utf-8') as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Auto-generated by get_oauth_token.py\n")
        f.write(f"export GLOBUS_ACCESS_TOKEN='{transfer_tokens['access_token']}'\n")
        if collection_tokens:
            f.write(
                "export GLOBUS_HTTPS_ACCESS_TOKEN="
                f"'{collection_tokens['access_token']}'\n"
            )
        f.write(f"export GLOBUS_COLLECTION_ID='{collection_id}'\n")
    os.chmod(sh_path, 0o700)

    print(f"Token saved to: {txt_path}")
    print(f"JSON saved to: {json_path}")
    print(f"Source-able script saved to: {sh_path}")
    print(f"Load it with: source {sh_path}")

    return {
        'transfer': transfer_tokens['access_token'],
        'collection': collection_tokens['access_token'] if collection_tokens else None,
        'refresh': transfer_tokens.get('refresh_token')
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Get Globus OAuth tokens for Transfer + collection scopes."
    )
    parser.add_argument(
        "collection_id",
        nargs="?",
        default="6c54cade-bde5-45c1-bdea-f4bd71dba2cc", # Globus Tutorial Collection 1 (https://app.globus.org/file-manager?origin_id=6c54cade-bde5-45c1-bdea-f4bd71dba2cc&origin_path=%2F)
        help="Collection/endpoint ID to request HTTPS + data_access scopes for.",
    )
    parser.add_argument(
        "--client-id",
        required=True,
        help="Native app client ID registered in Globus.",
    )
    parser.add_argument(
        "--auth-code",
        default=None,
        help="Authorization code from browser; loads saved PKCE state from /tmp/globus_pkce_state.json.",
    )
    parser.add_argument(
        "--with-data-access",
        action="store_true",
        help="Always request data_access scope (for non-HA mapped collections; "
             "omit for guest collections to avoid 'unknown scope' errors).",
    )
    return parser.parse_args()

def exchange_with_saved_state(auth_code):
    """Exchange auth code using PKCE state saved by a prior run."""
    state_path = "/tmp/globus_pkce_state.json"
    if not os.path.exists(state_path):
        print(f"ERROR: No saved PKCE state at {state_path}. Run without --auth-code first.", file=sys.stderr)
        sys.exit(1)
    with open(state_path) as f:
        state = json.load(f)
    client = globus_sdk.NativeAppAuthClient(state["client_id"])
    client.oauth2_start_flow(requested_scopes=state["scopes"])
    # Restore the verifier from the saved state
    client.current_oauth2_flow_manager.verifier = state["verifier"]
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
    return token_response, state["collection_id"]


if __name__ == "__main__":
    args = parse_args()

    try:
        if args.auth_code:
            token_response, collection_id = exchange_with_saved_state(args.auth_code)
            transfer_tokens = token_response.by_resource_server['transfer.api.globus.org']
            collection_tokens = token_response.by_resource_server.get(collection_id)
            print("=== Tokens Retrieved ===")
            print(f"\nTransfer API Access Token:\n{transfer_tokens['access_token']}\n")
            if collection_tokens:
                print(f"Collection Access Token:\n{collection_tokens['access_token']}\n")
            sh_path = "/tmp/globus_tokens.sh"
            with open(sh_path, "w") as f:
                f.write("#!/usr/bin/env bash\n")
                f.write(f"export GLOBUS_ACCESS_TOKEN='{transfer_tokens['access_token']}'\n")
                if collection_tokens:
                    f.write(f"export GLOBUS_HTTPS_ACCESS_TOKEN='{collection_tokens['access_token']}'\n")
                f.write(f"export GLOBUS_COLLECTION_ID='{collection_id}'\n")
            os.chmod(sh_path, 0o700)
            print(f"Tokens saved to {sh_path}")
            print(f"Load with: source {sh_path}")
        else:
            tokens = get_tokens_for_collection(
                args.collection_id, args.client_id, force_data_access=args.with_data_access
            )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
