from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Request, Depends
from fastapi.responses import JSONResponse
import uvicorn
from markitdown import MarkItDown
from pathlib import Path
import time
from fastapi.middleware.cors import CORSMiddleware
from config import settings
import logging
from prompt_builder import PromptBuilder
from schema_validator import validate_resume_data
from fastapi.security import APIKeyHeader
from openai import OpenAI

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Parser API",
    description="API for parsing PDF resumes using OpenAI and MarkItDown",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(403, "Invalid API key")
    return api_key


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/v1/parse/pdf")
async def parse_pdf(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
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

    dest = UPLOAD_DIR / file.filename
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

            # get the number of tokens in the parsed markdown
            input_tokens = len(parsed.text_content)

            if input_tokens > settings.MAX_FILE_TOKENS:
                logger.error(f"Too many tokens: {input_tokens}")
                raise HTTPException(400, "Too many tokens")

    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(500, f"Error parsing file")

    finally:
        dest.unlink()

    time_to_parse = time.time() - start_time
    llm_start_time = time.time()
    # parse the markdown using the prompt builder and validate the response
    try:
        prompt_builder = PromptBuilder()

        # get the structured data from LLM
        structured_data = prompt_builder.get_structured_data({"markdown": parsed.text_content})

        llm_response_time = time.time() - llm_start_time

        try:
            # validate the structured data
            validated_data = validate_resume_data(structured_data)

            # get the length of the validated data
            len_validated_data = len(validated_data.model_dump_json())

            # return the validated data
            return JSONResponse({
                "success": True,
                "data": validated_data.model_dump(mode='json'),
                "metadata": {
                    "file_size": file.size,
                    "model": settings.OPENAI_MODEL,
                    "time_to_parse": time_to_parse, # in seconds
                    "llm_response_time": llm_response_time, # in seconds
                    "total_time": time.time() - start_time, # in seconds
                    "input_tokens": input_tokens,
                    "output_tokens": len_validated_data,
                }
            })
        except Exception as validation_error:
            # Return validation errors with 400 status
            logger.error(f"Validation error: {validation_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid resume data format: {str(validation_error)}"
            )

    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(500, f"Error parsing resume: {str(e)}")


@app.get("/v1/health")
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
        return {"status": "healthy", "openai": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "openai": "disconnected"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")