from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Security,
    Request,
    Depends,
    status,
)
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
from markitdown import MarkItDown
from pathlib import Path
from uuid import uuid4
import time
from fastapi.middleware.cors import CORSMiddleware
from config import settings
import logging
from prompt_builder import PromptBuilder
from schema_validator import validate_resume_data
from fastapi.security import APIKeyHeader
from openai import OpenAI
import tiktoken
from models import (
    ParsePdfResponse,
    ErrorResponse,
    Error,
    ResponseMetadata,
    HealthResponse,
)

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Parser API",
    description=(
        "Parse PDF/DOCX resumes into structured JSON using MarkItDown for extraction and"
        " OpenAI for schema-friendly normalization.\n\n"
        "Testing: You can use the header X-API-Key with the test key "
        "'1lazk6WGUWua3twKFSKdn65/cNVijyGyHFQgHsyC0fQ=' to try the API."
    ),
    version="0.1.0",
    contact={
        "name": "by Marcelo Lopes",
        "url": "https://github.com/marceloakalopes",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {"name": "Health", "description": "Service and dependency health"},
        {"name": "Parsing", "description": "Resume parsing operations"},
    ],
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "defaultModelExpandDepth": 2,
        "defaultModelRendering": "model",
        "displayRequestDuration": True,
        "docExpansion": "list",
        "tagsSorter": "alpha",
        "operationsSorter": "alpha",
        "filter": True,
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

API_KEY_HEADER = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the API key

    Args:
        api_key: The API key to verify

    Returns:
        The API key if it is valid

    Raises:
        HTTPException: If the API key is invalid
    """

    if settings.MODE == "showcase":
        return api_key

    # Accept either a single API_KEY or any from API_KEYS
    configured_keys = set(filter(None, [settings.API_KEY, *settings.API_KEYS]))
    if not configured_keys:
        raise HTTPException(403, "API key not configured")
    if not api_key or api_key not in configured_keys:
        raise HTTPException(403, "Invalid API key")
    return api_key


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Return uniform ErrorResponse for all HTTPExceptions."""
    status_code = exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR
    generic_code = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        500: "internal_error",
    }.get(status_code, "error")

    error = Error(code=generic_code, message=str(exc.detail))
    body = ErrorResponse(success=False, error=error)
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """Handle FastAPI request validation errors (422)."""
    error = Error(
        code="validation_error",
        message="Invalid request parameters",
        details={"errors": exc.errors()},
    )
    body = ErrorResponse(success=False, error=error)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body.model_dump()
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors."""
    logger.exception("Unhandled error: %s", exc)
    error = Error(code="internal_error", message="An unexpected error occurred")
    body = ErrorResponse(success=False, error=error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.model_dump()
    )


@app.get("/")
def read_root():
    return {"message": "Visit https://api.profy.me/docs to learn more about the API"}


@app.post(
    "/v1/parse/pdf",
    response_model=ParsePdfResponse,
    status_code=status.HTTP_200_OK,
    tags=["Parsing"],
    summary="Parse a PDF resume into structured JSON",
    description=(
        "Uploads a PDF resume and returns structured resume data. The file is parsed"
        " with MarkItDown and normalized using an LLM. Size and token limits apply."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Bad Request: invalid file, size, tokens, or validation errors",
        },
        403: {
            "model": ErrorResponse,
            "description": "Forbidden: invalid API key",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error: processing/parsing failure",
        },
    },
)
async def parse_pdf(
    file: UploadFile = File(..., description="PDF file to parse (application/pdf)"),
    api_key: str = Depends(verify_api_key),
):
    start_time = time.time()

    # initialize the markitdown parser
    markitdown = MarkItDown(
        model=settings.OPENAI_MODEL,
        # temperature=0.0,
        # max_tokens=10000,
        # top_p=1.0,
        # frequency_penalty=0.0,
    )

    # check if file is provided
    if not file.filename:
        logger.error(f"No file name provided")
        raise HTTPException(400, "No file name provided")

    # check if file is a pdf
    if not file.filename.lower().endswith(".pdf"):
        logger.error(f"Only PDF files are allowed: {file.filename}")
        raise HTTPException(400, "Only PDF files are allowed")

    # check if file is too large
    if file.size is None or file.size > settings.MAX_FILE_SIZE:
        logger.error(f"File is too large: {file.size}")
        raise HTTPException(400, "File is too large")

    # generate a unique filename to avoid collisions
    unique_name = f"{uuid4()}.pdf"
    dest = UPLOAD_DIR / unique_name
    try:
        with dest.open("wb") as buffer:
            # read in 1 MB chuncks
            while True:
                chunck = await file.read(1024 * 1024)
                if not chunck:
                    break
                buffer.write(chunck)

            # parse the pdf using pymupdf4llm and get the markdown
            parsed = markitdown.convert(dest)

            # get the number of tokens in the parsed markdown (true token count)
            try:
                encoding = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(parsed.text_content))

            if input_tokens > settings.MAX_FILE_TOKENS:
                logger.error(f"Too many tokens: {input_tokens}")
                raise HTTPException(400, "Too many tokens")

    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(500, "Error parsing file")

    finally:
        # ensure temp file is removed without crashing if already gone
        dest.unlink(missing_ok=True)

    time_to_parse = time.time() - start_time
    # parse the markdown using the prompt builder and validate the response
    try:
        prompt_builder = PromptBuilder()

        # time llm response
        llm_response_start_time = time.time()

        # get the structured data from LLM
        structured_data = prompt_builder.get_structured_data(
            {"markdown": parsed.text_content}
        )

        llm_response_time = time.time() - llm_response_start_time

        try:
            # validate the structured data
            validated_data = validate_resume_data(structured_data)

            # compute output tokens using the model's tokenizer
            try:
                encoding = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            output_tokens = len(encoding.encode(validated_data.model_dump_json()))

            # return the validated data
            response = ParsePdfResponse(
                success=True,
                data=validated_data,
                metadata=ResponseMetadata(
                    file_size=file.size,
                    model=settings.OPENAI_MODEL,
                    time_to_parse=time_to_parse,
                    llm_response_time=llm_response_time,
                    total_time=time.time() - start_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )
            return JSONResponse(content=response.model_dump())
        except Exception as validation_error:
            # Return validation errors with 400 status
            logger.error(f"Validation error: {validation_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid resume data format: {str(validation_error)}",
            )

    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(500, f"Error parsing resume: {str(e)}")


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health and OpenAI connectivity",
    description="Returns overall health status and whether the OpenAI API is reachable.",
    responses={
        403: {"model": ErrorResponse, "description": "Forbidden: invalid API key"},
    },
)
async def health_check(api_key: str = Depends(verify_api_key)):
    """
    Check the health of the API

    Returns:
        A JSON response with the status of the API

    Raises:
        HTTPException: If the API is unhealthy
    """
    openai_client = OpenAI()
    try:
        openai_client.models.list()
        return {"success": True, "status": "healthy", "openai": "connected"}
    except Exception as e:
        err = Error(code="openai_unavailable", message=str(e))
        return {
            "success": False,
            "status": "unhealthy",
            "openai": "disconnected",
            "error": err.model_dump(),
        }


@app.post(
    "/v1/parse/docx",
    response_model=ParsePdfResponse,
    status_code=status.HTTP_200_OK,
    tags=["Parsing"],
    summary="Parse a DOCX resume into structured JSON",
    description=(
        "Uploads a DOCX resume and returns structured resume data. The file is parsed"
        " with MarkItDown and normalized using an LLM. Size and token limits apply."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Bad Request: invalid file, size, tokens, or validation errors",
        },
        403: {
            "model": ErrorResponse,
            "description": "Forbidden: invalid API key",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error: processing/parsing failure",
        },
    },
)
async def parse_docx(
    file: UploadFile = File(
        ...,
        description="DOCX file to parse (application/vnd.openxmlformats-officedocument.wordprocessingml.document)",
    ),
    api_key: str = Depends(verify_api_key),
):
    start_time = time.time()

    markitdown = MarkItDown(
        model=settings.OPENAI_MODEL,
        embed_images=False,
        ocr=False,
        max_chars=120_000,
    )

    if not file.filename:
        logger.error(f"No file name provided")
        raise HTTPException(400, "No file name provided")

    if not file.filename.lower().endswith(".docx"):
        logger.error(f"Only DOCX files are allowed: {file.filename}")
        raise HTTPException(400, "Only DOCX files are allowed")

    if file.size is None or file.size > settings.MAX_FILE_SIZE:
        logger.error(f"File is too large: {file.size}")
        raise HTTPException(400, "File is too large")

    unique_name = f"{uuid4()}.docx"
    dest = UPLOAD_DIR / unique_name
    try:
        with dest.open("wb") as buffer:
            while True:
                chunck = await file.read(1024 * 1024)
                if not chunck:
                    break
                buffer.write(chunck)

            parsed = markitdown.convert(dest)

            try:
                encoding = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(parsed.text_content))

            if input_tokens > settings.MAX_FILE_TOKENS:
                logger.error(f"Too many tokens: {input_tokens}")
                raise HTTPException(400, "Too many tokens")

    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(500, "Error parsing file")
    finally:
        dest.unlink(missing_ok=True)

    time_to_parse = time.time() - start_time
    try:
        prompt_builder = PromptBuilder()

        llm_response_start_time = time.time()
        structured_data = prompt_builder.get_structured_data(
            {"markdown": parsed.text_content}
        )
        llm_response_time = time.time() - llm_response_start_time

        try:
            validated_data = validate_resume_data(structured_data)

            try:
                encoding = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
            output_tokens = len(encoding.encode(validated_data.model_dump_json()))

            response = ParsePdfResponse(
                success=True,
                data=validated_data,
                metadata=ResponseMetadata(
                    file_size=file.size,
                    model=settings.OPENAI_MODEL,
                    time_to_parse=time_to_parse,
                    llm_response_time=llm_response_time,
                    total_time=time.time() - start_time,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )
            return JSONResponse(content=response.model_dump())
        except Exception as validation_error:
            logger.error(f"Validation error: {validation_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid resume data format: {str(validation_error)}",
            )
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(500, f"Error parsing resume: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", workers=1, reload=True)
