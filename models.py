from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from schema_validator import ResumeData


class ResponseMetadata(BaseModel):
    """Additional metadata returned with successful or error responses."""

    file_size: Optional[int] = Field(
        default=None, description="Uploaded file size in bytes"
    )
    model: Optional[str] = Field(default=None, description="Model identifier used")
    time_to_parse: Optional[float] = Field(
        default=None, description="Seconds spent parsing the PDF"
    )
    llm_response_time: Optional[float] = Field(
        default=None, description="Seconds spent generating LLM response"
    )
    total_time: Optional[float] = Field(
        default=None, description="Total request processing time in seconds"
    )
    input_tokens: Optional[int] = Field(
        default=None, description="Estimated number of input tokens"
    )
    output_tokens: Optional[int] = Field(
        default=None, description="Estimated number of output tokens"
    )

    model_config = {
        "title": "Core/ResponseMetadata",
        "json_schema_extra": {
            "examples": [
                {
                    "file_size": 123456,
                    "model": "gpt-4.1-nano",
                    "time_to_parse": 1.23,
                    "llm_response_time": 0.85,
                    "total_time": 2.08,
                    "input_tokens": 1024,
                    "output_tokens": 256,
                }
            ]
        }
    }


class Error(BaseModel):
    """Error details formatted for clients and OpenAPI documentation."""

    code: str = Field(
        description="Stable machine-readable error code (e.g., 'invalid_file_type')"
    )
    message: str = Field(
        description="Human-readable description of the error suitable for logs and UI"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured details to aid debugging"
    )

    model_config = {
        "title": "Core/Error",
        "json_schema_extra": {
            "examples": [
                {
                    "code": "invalid_api_key",
                    "message": "Invalid API key",
                },
                {
                    "code": "validation_error",
                    "message": "Invalid resume data format",
                    "details": {"field": "experience[0].startDate"},
                },
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error envelope returned on non-2xx responses."""

    success: bool = Field(default=False, description="Indicates request failed")
    error: Error
    metadata: Optional[ResponseMetadata] = Field(
        default=None, description="Optional metadata for troubleshooting"
    )

    model_config = {
        "title": "Core/ErrorResponse",
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": {"code": "invalid_file_type", "message": "Only PDF files are allowed"},
                }
            ]
        }
    }


class ParsePdfResponse(BaseModel):
    """Successful response for the parse PDF endpoint."""

    success: bool = Field(default=True, description="Indicates request succeeded")
    data: ResumeData
    metadata: ResponseMetadata

    model_config = {
        "title": "Parsing/ParsePdfResponse",
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "data": {
                        "id": 1,
                        "username": "janedoe",
                        "name": "Jane Doe",
                        "headline": "Senior Software Engineer",
                        "experience": [],
                        "education": [],
                        "projects": [],
                        "skills": ["Python", "FastAPI"],
                        "socials": [],
                    },
                    "metadata": {
                        "file_size": 234567,
                        "model": "gpt-4.1-nano",
                        "time_to_parse": 1.11,
                        "llm_response_time": 0.92,
                        "total_time": 2.45,
                        "input_tokens": 980,
                        "output_tokens": 210,
                    },
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    success: bool
    status: str = Field(description="'healthy' or 'unhealthy'")
    openai: str = Field(description="'connected' or 'disconnected'")
    error: Optional[Error] = None

    model_config = {
        "title": "Health/HealthResponse",
        "json_schema_extra": {
            "examples": [
                {"success": True, "status": "healthy", "openai": "connected"},
                {
                    "success": False,
                    "status": "unhealthy",
                    "openai": "disconnected",
                    "error": {
                        "code": "openai_unavailable",
                        "message": "Could not reach OpenAI API",
                    },
                },
            ]
        }
    }


