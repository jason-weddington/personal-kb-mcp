"""CLI entry point for the standalone KB Explorer web server."""

import threading
import webbrowser


def main() -> None:
    """Run the KB Explorer web server and open browser."""
    import uvicorn

    from personal_kb.web.app import create_app

    app = create_app()

    # Open browser after a short delay
    def _open_browser() -> None:
        import time

        time.sleep(1)
        webbrowser.open("http://localhost:8765")

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
