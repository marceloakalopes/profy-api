import json
from pathlib import Path
from typing import Dict
import os
from openai import OpenAI
from fastapi import HTTPException
import tiktoken


class PromptBuilder:
    """Handles OpenAI prompt construction and interaction."""

    def __init__(self):
        """Initialize OpenAI client and load schema."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

        # Load schema
        schema_path = Path(__file__).parent / "schema.json"
        with open(schema_path, "r") as f:
            self.schema = json.load(f)

    def _build_system_prompt(self) -> str:
        """
        Build system prompt with schema instructions.

        Returns:
            str: Formatted system prompt
        """
        return f"""You are a precise resume parser that extracts structured data from resume text.
        You will receive resume content in Markdown format, which preserves the original document structure.
        
        DOCUMENT STRUCTURE GUIDELINES:
        - Section headers are marked with # (main) or ## (subsection)
        - Bold text is marked with **
        - Italic text is marked with *
        - Section breaks are marked with ---
        
        VALIDATION RULES:
        1. Dates:
           - All dates MUST be in YYYY-MM-DD format
           - End dates cannot be before start dates
           - Future dates are not allowed
           - Use "Present" or current date for ongoing positions
        
        2. Required Fields:
           - Generate random integer (1-100000) for id
           - Leave username field empty (null)
           - Leave imageUrl field empty (null)
           - If no headline found, generate a concise professional headline (MAX 50 chars) based on experience
           - If no about found, generate a professional summary (MAX 250 chars) based on experience and skills
           - All array fields (experience, education, projects, skills, socials) must have at least one item
           - Experience dates must not overlap for same company
        
        3. Field-Specific Rules:
           - Experience type must be one of: full-time, part-time, freelance, contract, internship, volunteer
           - Skills should be individual technologies/competencies, not descriptions
           - Social URLs must be valid and include platform name
           - Project links must be valid URLs
        
        EXAMPLE PARSING:
        Input Markdown:
        ```
        # John Doe
        **Senior Software Engineer**
        
        ## Experience
        **Software Engineer** at *TechCorp*
        2020-01 to Present
        ```
        
        Expected Output:
        {{
            "id": 12345,
            "username": null,
            "imageUrl": null,
            "name": "John Doe",
            "headline": "Senior Software Engineer",
            "about": "Experienced Software Engineer with a proven track record in...",
            "experience": [{{
                "company": "TechCorp",
                "title": "Software Engineer",
                "startDate": "2020-01-01",
                "endDate": "2024-03-21",
                "type": "full-time",
                "location": "Not Specified"
            }}]
            // ... other required fields with appropriate defaults
        }}
        
        SCHEMA DEFINITION:
        {json.dumps(self.schema["parameters"], indent=2)}
        
        FAILURE CASES TO AVOID:
        1. Incomplete dates (e.g., "2020" instead of "2020-01-01")
        2. Missing required fields (provide reasonable defaults if not in resume)
        3. Invalid URLs in project links or social profiles
        4. Inconsistent date ranges or overlapping experiences
        5. Skills that are phrases instead of individual technologies
        
        Return ONLY valid JSON matching the schema. No additional text or explanations."""

    def get_structured_data(self, resume_data: Dict[str, str]) -> Dict[str, str]:
        """
        Send resume text to OpenAI and get structured data.

        Args:
            resume_data (Dict[str, str]): Resume data in markdown and text formats

        Returns:
            Dict: Structured data matching the schema

        Raises:
            HTTPException: If the resume is too long to process
        """
        # Get the encoding for the model
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base encoding (used by gpt-4 and gpt-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = f"""Parse this resume:

        {resume_data['markdown']}"""

        # Count tokens
        system_prompt_tokens = len(encoding.encode(system_prompt))
        user_prompt_tokens = len(encoding.encode(user_prompt))

        # Set limits based on model and account for expected response tokens
        if self.model.startswith("gpt-4"):
            TOKEN_LIMIT = 6000  # Leave ~2K tokens for response in 8K context
        else:
            TOKEN_LIMIT = 3000  # Leave ~1K tokens for response in 4K context

        total_tokens = system_prompt_tokens + user_prompt_tokens

        if total_tokens > TOKEN_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=f"Resume is too long to process. Current tokens: {total_tokens}, Limit: {TOKEN_LIMIT}",
            )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            return json.loads(completion.choices[0].message.content or "{}")

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing resume with OpenAI: {str(e)}"
            )
