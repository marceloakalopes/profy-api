## 📑 Resume Parser API

FastAPI service that converts PDF and DOCX resumes into structured JSON using MarkItDown for text extraction and OpenAI for normalization. It exposes a small set of authenticated endpoints and comprehensive OpenAPI/Swagger docs.

### Features

- **PDF parsing** via MarkItDown
- **LLM-powered normalization** to a stable schema
- **Typed responses** with Pydantic models and examples
- **Uniform error envelope** across all endpoints
- **CORS** and **API key** authentication
- **Detailed Swagger** docs with examples and error models
- **Performance metrics** in response metadata

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Installation

```bash
git clone <repository-url>
cd profy-api
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and adjust values. All settings are optional unless noted:

```env
### OpenAI ###
OPENAI_API_KEY=                         # Your OpenAI API key. Required to call OpenAI.
OPENAI_MODEL=gpt-4.1-nano              # Model used by MarkItDown/OpenAI.
OPENAI_MAX_TOKENS=7000                 # Max tokens for LLM (upper bound; app may use less).

### Security ###
ALLOWED_ORIGINS=["*"]                  # CORS: list of allowed origins. "*" allows all (dev-friendly).

### API KEY ###
API_KEYS=["dev-key-1","dev-key-2"]   # Array of accepted API keys.
API_KEY=                                # Optional legacy single key; still supported.

### Mode ###
MODE=showcase                           # 'showcase' skips API key enforcement; set to other value for prod.
```

### Running Locally

```bash
python main.py
```

The API runs at `http://127.0.0.1:8000`. Visit Swagger UI at `http://127.0.0.1:8000/docs` and the OpenAPI JSON at `http://127.0.0.1:8000/openapi.json`.

## Authentication

- Header: `X-API-Key: <your key>`
- In `MODE=showcase`, API key validation is skipped for convenience.
- For testing, you can use this API key: `1lazk6WGUWua3twKFSKdn65/cNVijyGyHFQgHsyC0fQ=`

## Endpoints

### Health Check

```http
GET /v1/health
X-API-Key: <your key>
```

Example success response:

```json
{
  "success": true,
  "status": "healthy",
  "openai": "connected"
}
```

Example error response:

```json
{
  "success": false,
  "status": "unhealthy",
  "openai": "disconnected",
  "error": { "code": "openai_unavailable", "message": "..." }
}
```

### Parse Resume (PDF)

```http
POST /v1/parse/pdf
X-API-Key: <your key>
Content-Type: multipart/form-data
Body: file=@resume.pdf
```

Success response:

```json
{
  "success": true,
  "data": { /* ResumeData fields */ },
  "metadata": {
    "file_size": 123456,
    "model": "gpt-4.1-nano",
    "time_to_parse": 1.23,
    "llm_response_time": 0.85,
    "total_time": 2.08,
    "input_tokens": 1024,
    "output_tokens": 256
  }
}
```

### Parse Resume (DOCX)

```http
POST /v1/parse/docx
X-API-Key: <your key>
Content-Type: multipart/form-data
Body: file=@resume.docx
```

Response shape is identical to the PDF endpoint.

Error response (uniform across endpoints):

```json
{
  "success": false,
  "error": {
    "code": "invalid_file_type",
    "message": "Only PDF files are allowed"
  }
}
```

## Data Models

- `ResumeData`: Structured resume output (see `schema_validator.py` / `schema.json`).
- `ParsePdfResponse`: Success envelope with `data` and `metadata`.
- `ErrorResponse`: Error envelope with stable `code` and `message`.

OpenAPI includes embedded examples for these models in the Swagger UI.

## Development

- Start the server: `python main.py`
- Linting: Pydantic validation errors return a 422 with structured details.
- CORS: Configure allowed origins via `ALLOWED_ORIGINS`.

## Contribution Guidelines

1. Fork and create a feature branch.
2. Write clear commits and add tests if applicable.
3. Ensure Swagger docs remain descriptive: update response models, tags, and examples when changing endpoints.
4. Open a PR with a summary of changes and screenshots of updated docs.
