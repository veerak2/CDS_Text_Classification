from pydantic import BaseModel

class Text(BaseModel):
    transcript: str