# Clinical assessment frontend

React + Vite interface for the FastAPI inference service.

From this directory:

```bash
npm install
npm run dev
```

The Vite development server proxies `/health`, `/symptoms`, and `/predict` to `http://127.0.0.1:8000`. Start the API from the repository root with `uvicorn api.main:app --reload`.

For a separately hosted API, set `VITE_API_BASE_URL` to its base URL before building. The API host must permit browser requests from the frontend origin.
