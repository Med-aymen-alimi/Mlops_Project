from pydantic import BaseModel, Field
from typing import Optional, List

class TransactionModel(BaseModel):
    movies: str = Field(..., description="Title of the movie")
    year: str = Field(..., description="Release year of the movie")
    genre: str = Field(..., description="Movie genres (comma-separated)")
    #rating: float = Field(..., ge=0, le=10, description="Movie rating (0-10)")
    one_line: str = Field(..., alias="one_line", description="Brief description of the movie")
    stars: str = Field(..., description="Director and stars information")
    votes: Optional[float] = Field(None, description="Number of votes")
    runtime: Optional[float] = Field(None, description="Runtime in minutes")  # Updated type
    gross: Optional[str] = Field(None, description="Gross earnings")

    class Config:
        json_schema_extra = {
            "example": 
            { "movies": "vikings",
            "one_line": "Synopsis",
            "year": "2019",
            "runtime": 100,
            "genre": "action,drama,adventure", 
            "stars": "Director: Peter Thorwarth | Stars: Peri Baumeister, Carl Anton Koch", 
            "votes": 1000, 
             "gross": "100M" }
        }
        allow_population_by_field_name = True