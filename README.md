# Resume Parser API

A FastAPI-based REST API that parses PDF resumes using OpenAI and MarkItDown to extract structured data.

## Features

- PDF resume parsing and text extraction
- OpenAI-powered structured data extraction
- JSON schema validation for parsed data
- API key authentication
- CORS support
- Request processing time tracking
- Health check endpoint
- File size and token count limitations
- Comprehensive error handling and logging

## Prerequisites

- Python 3.x
- OpenAI API key
- FastAPI
- uvicorn
- Other dependencies (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd profy-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file):
```env
API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4  # or your preferred model
MAX_FILE_SIZE=10000000  # in bytes
MAX_FILE_TOKENS=10000
ALLOWED_ORIGINS=["http://localhost:3000"]  # adjust as needed
```

## API Endpoints

### Health Check
```http
GET /v1/health
Header: X-API-Key: your_api_key
```
Returns the health status of the API and OpenAI connection.

### Parse Resume
```http
POST /v1/parse/pdf
Header: X-API-Key: your_api_key
Content-Type: multipart/form-data
Body: file=@resume.pdf
```
Parses a PDF resume and returns structured data.

#### Response Format
```json
{
    "success": true,
    "data": {
        // Structured resume data based on schema
    },
    "metadata": {
        "file_size": 123456,
        "model": "gpt-4",
        "time_to_parse": 1.23,
        "llm_response_time": 2.34,
        "total_time": 3.57,
        "input_tokens": 1000,
        "output_tokens": 500
    }
}
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid API keys (403)
- Invalid file formats (400)
- File size limits (400)
- Token count limits (400)
- Schema validation errors (400)
- Processing errors (500)

## Development

To run the server in development mode:

```bash
python main.py
```

The server will start at `http://0.0.0.0:8000`.

## Architecture

The project consists of several key components:
- `main.py`: FastAPI application and route handlers
- `config.py`: Configuration settings and environment variables
- `prompt_builder.py`: OpenAI prompt construction and handling
- `schema_validator.py`: JSON schema validation for resume data
- `schema.json`: Resume data structure definition

## Performance

The API includes performance monitoring:
- Request processing time tracking
- Token usage monitoring
- File size validation
- Response time metrics

## Security

- API key authentication required for all endpoints
- CORS middleware with configurable origins
- File size limitations
- Token count restrictions
- Temporary file handling with automatic cleanup

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 