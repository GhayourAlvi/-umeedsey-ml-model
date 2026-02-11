from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PatientRecord(BaseModel):
    """Flexible patient record model that accepts any fields."""

    class Config:
        extra = "allow"
        json_schema_extra = {"example": {}}


class PredictionResponse(BaseModel):
    predictionModeofdeliverywithout: str
    predictionAntinatalwithout: str
    predictionNeonatalwithout: str


class ModeDeliveryResponse(BaseModel):
    prediction: str


class AntenatalResponse(BaseModel):
    prediction: str


class NeonatalResponse(BaseModel):
    prediction: str


class PostnatalResponse(BaseModel):
    prediction: str


class PatientRecordWithId(PatientRecord):
    patient_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

