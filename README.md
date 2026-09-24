# Document Ingestion API

- Install the dependencies with `pip install -r requirements.txt`.
- Copy `.env.example` to `.env` and add the required Pinecone, OpenRouter, Redis, and database settings.
- Start the API with `uvicorn app.main:app --reload`.
- Open `http://127.0.0.1:8000/docs` to use the interactive API documentation.
- The API supports document ingestion, conversational chat, and interview booking management under `/api/v1`.
- Check `http://127.0.0.1:8000/health` to confirm that the application is running.
- For document id, check the app.db which is local sql.