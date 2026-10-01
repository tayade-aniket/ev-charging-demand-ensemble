"""
Pydantic Schemas for Request Validation and Response Serialization.

Provides input validation for EV charging session attributes, ensuring all
numerical boundaries and categorical values conform to domain specifications.

Author: Academic Machine Learning Mentorship & Research Team
Project: EV Charging Demand Prediction Using Ensemble Machine Learning
"""

from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


ChargerTypeEnum = Literal["DC Fast Charger", "Level 1", "Level 2"]
TimeOfDayEnum = Literal["Morning", "Afternoon", "Evening", "Night"]
DayOfWeekEnum = Literal[
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]
UserTypeEnum = Literal["Commuter", "Casual Driver", "Long-Distance Traveler"]
VehicleModelEnum = Literal[
    "BMW i3", "Hyundai Kona", "Chevy Bolt", "Nissan Leaf", "Tesla Model 3"
]
LocationEnum = Literal["Chicago", "Houston", "Los Angeles", "New York", "San Francisco"]


class ChargingSessionInput(BaseModel):
    """
    Input schema representing the pre-session arrival attributes of an EV.
    All fields are validated to prevent out-of-distribution or malformed inputs.
    """
    battery_capacity: float = Field(
        ...,
        gt=0.0,
        le=300.0,
        description="Total battery capacity in kWh (must be > 0 and <= 300).",
        json_schema_extra={"example": 60.0}
    )
    soc_start: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="State of Charge (Start %) upon arrival (0.0% to 100.0%).",
        json_schema_extra={"example": 25.0}
    )
    vehicle_age: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Vehicle age in years (0.0 to 30.0).",
        json_schema_extra={"example": 3.0}
    )
    temperature: float = Field(
        ...,
        ge=-50.0,
        le=65.0,
        description="Ambient temperature in degrees Celsius (-50.0 to 65.0).",
        json_schema_extra={"example": 28.0}
    )
    distance_driven: float = Field(
        ...,
        ge=0.0,
        le=2000.0,
        description="Distance driven since previous charging session in km.",
        json_schema_extra={"example": 120.0}
    )
    charger_type: ChargerTypeEnum = Field(
        ...,
        description="Type of charger used ('DC Fast Charger', 'Level 1', 'Level 2').",
        json_schema_extra={"example": "DC Fast Charger"}
    )
    time_of_day: TimeOfDayEnum = Field(
        ...,
        description="Time period ('Morning', 'Afternoon', 'Evening', 'Night').",
        json_schema_extra={"example": "Evening"}
    )
    day_of_week: DayOfWeekEnum = Field(
        ...,
        description="Day of week ('Monday' through 'Sunday').",
        json_schema_extra={"example": "Saturday"}
    )
    user_type: UserTypeEnum = Field(
        ...,
        description="Category of EV user ('Commuter', 'Casual Driver', 'Long-Distance Traveler').",
        json_schema_extra={"example": "Commuter"}
    )
    vehicle_model: Optional[VehicleModelEnum] = Field(
        default="Tesla Model 3",
        description="Vehicle model name.",
        json_schema_extra={"example": "Tesla Model 3"}
    )
    location: Optional[LocationEnum] = Field(
        default="Chicago",
        description="Charging station city location.",
        json_schema_extra={"example": "Chicago"}
    )

    @field_validator('soc_start')
    @classmethod
    def validate_soc(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError("State of Charge (Start %) must be strictly between 0% and 100%.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "battery_capacity": 60.0,
                "soc_start": 25.0,
                "vehicle_age": 3.0,
                "temperature": 28.0,
                "distance_driven": 120.0,
                "charger_type": "DC Fast Charger",
                "time_of_day": "Evening",
                "day_of_week": "Saturday",
                "user_type": "Commuter",
                "vehicle_model": "Tesla Model 3",
                "location": "Chicago"
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Output schema returning predicted energy consumption and metadata.
    """
    predicted_energy_kwh: float = Field(
        ...,
        description="Expected energy demand in kilowatt-hours (kWh) for the session."
    )
    unit: str = Field(
        default="kWh",
        description="Measurement unit for electrical energy."
    )
    model_name: str = Field(
        default="Stacking Regressor (LinearRegression + DecisionTree + KNN -> Ridge)",
        description="Ensemble architecture utilized for inference."
    )
    status: str = Field(default="success")


class HealthResponse(BaseModel):
    """
    System health check response schema.
    """
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=True)


class ProjectInfoResponse(BaseModel):
    """
    Root endpoint response schema providing project metadata and documentation links.
    """
    title: str
    description: str
    authors: str
    version: str
    architecture: str
    target_variable: str
    best_model_name: str
    best_model_metrics: Dict[str, Any]
    endpoints: Dict[str, str]
