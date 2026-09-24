# Existing prototype deployment

Scope: preserve React/Vite, FastAPI, C++ ONNX engine and trained models.

1. Test and add configurable API origin, plot URLs, CORS and health alias.
2. Package actual artifacts into the Docker image; non-root startup and PORT.
3. Add Vercel configuration and free CPU Docker Space upload command.
4. Run API/frontend tests, production build and container verification if Docker permits.
5. Authenticate hosting accounts, publish backend, connect Vercel, test public upload.

Account credentials are currently absent. No final URLs may be claimed until
hosting returns them and a production health/prediction check passes.
