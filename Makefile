.PHONY: run

run:
	UVICORN_LOG_LEVEL=info uvicorn --app-dir src app.main:app --reload --port 8000
