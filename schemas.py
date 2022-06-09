from datetime import datetime, date
from pydantic import BaseModel



class cons(BaseModel):
    date_: date
    period: int
    
    class Config:
        schema_extra = {
            "example": {
                "date_": '2022-06-08',
                "period": 5
            }
        }
