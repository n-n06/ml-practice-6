import enum
from pydantic import BaseModel

class Job(enum.Enum):
    MANAGEMENT = "management"	
    BLUE_COLLAR = "blue-collar"	
    TECHNICIAN = "technician"	
    ADMIN = "admin"	
    SERVICES = "services"	
    RETIRED = "retired"	
    SELF_EMPLOYED = "self-employed"	
    STUDENT = "student"	
    UNEMPLOYED = "unemployed"	
    ENTREPRENEUR = "entrepreneur"	
    HOUSEMAID = "housemaid"	
    UNKNOWN = "unknown"	

class MaritalStatus(enum.Enum):
    MARRIED = "married"
    SINGEL = "single"
    DIVORCED = "divorced"

class Education(enum.Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNKNOWN = "unknown"

class BoolValue(enum.Enum):
    YES = "yes"
    NO = "no"

class ContactType(enum.Enum):
    TELEPHONE = "telephone"
    CELLULAR = "cellular"
    UNKNOWN = "unknown"

class OutcomeType(enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClientData(BaseModel):
    age: int
    job: Job
    marital: MaritalStatus
    education: Education
    default: BoolValue
    balance: int
    housing: BoolValue
    loan: BoolValue
    contact: ContactType
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: OutcomeType

