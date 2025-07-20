from datetime import datetime
from typing import List
from enum import Enum
from pydantic import BaseModel, HttpUrl
import json
from pathlib import Path

# Load schema from file
SCHEMA_PATH = Path(__file__).parent / "schema.json"
with open(SCHEMA_PATH, "r") as f:
    RESUME_SCHEMA = json.load(f)

class EmploymentType(str, Enum):
    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    FREELANCE = "freelance"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    VOLUNTEER = "volunteer"

class Experience(BaseModel):
    company: str | None = None
    title: str | None = None
    location: str | None = "Not specified"
    startDate: str | None = None
    endDate: str | None = None
    type: EmploymentType | None = EmploymentType.FULL_TIME

class Education(BaseModel):
    school: str | None = None
    degree: str | None = None
    startDate: str | None = None
    endDate: str | None = None

class Project(BaseModel):
    name: str | None = "Untitled Project"
    description: str | None = ""
    technologies: List[str] = []
    link: str | None = None

class Social(BaseModel):
    name: str
    url: str | None = None

class ResumeData(BaseModel):
    id: int | None = None
    username: str | None = None
    imageUrl: str | None = None
    name: str | None = "Anonymous"
    headline: str | None = ""
    about: str | None = ""
    experience: List[Experience] = []
    education: List[Education] = []
    projects: List[Project] = []
    skills: List[str] = []
    socials: List[Social] = []
    updated_at: datetime | None = None

class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    response_metadata: dict

class SuccessResponse(BaseModel):
    ok: bool = True
    data: ResumeData

def validate_resume_data(data: dict) -> ResumeData:
    """
    Validate resume data against the schema.
    
    Args:
        data (dict): Resume data to validate
        
    Returns:
        ResumeData: Validated resume data model
        
    Raises:
        ValidationError: If data doesn't match schema
    """
    return ResumeData(**data) 