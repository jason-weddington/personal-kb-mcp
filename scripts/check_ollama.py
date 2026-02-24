"""Quick check that Ollama is running and the embedding model is available."""

import sys

import httpx

from personal_kb.config import get_embedding_model, get_ollama_url


def main() -> None:
    """Check Ollama connectivity and embedding model availability."""
    url = get_ollama_url()
    model = get_embedding_model()
    print(f"Checking Ollama at {url} for model {model}...")

    try:
        resp = httpx.get(f"{url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"Available models: {', '.join(models) or '(none)'}")

        if any(model in m for m in models):
            print(f"  {model} is available")
        else:
            print(f"  {model} not found — run: ollama pull {model}")
            sys.exit(1)
    except httpx.ConnectError:
        print("  Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"  Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
