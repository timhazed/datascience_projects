from pydantic import BaseModel, Field

class ReviewAnalysis(BaseModel):
    """Model representing the sentiment analysis results for a specific review."""
    
    review_id: int
    aspect: str = Field(description="The aspect being evaluated")
    sentiment: str = Field(..., pattern="^(Positive|Negative)$", description="Sentiment related to the aspect, either Positive or Negative")
    polarity: float = Field(..., ge=0, le=1, description="Sentiment polarity, between 0 (extremely negative) and 1 (extremely positive)")

