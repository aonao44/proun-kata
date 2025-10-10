.PHONY: run

run:
	UVICORN_LOG_LEVEL=info uvicorn --app-dir src api.app:app --host 0.0.0.0 --reload --port ${UVICORN_PORT:-8000}
