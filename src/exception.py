import sys
from src.logging import get_logger

logging = get_logger(__name__)

class BankLoanDetection:
    def __init__(self,message=None,error=None):
        """
        Exception Configuration
        """
        super().__init__(message)
        self,errors=errors
        if message:
            "Log the Error Message"
            logging.error(message)
        if errors:
            "Log the Error Details"
            logging.error(error)

class CustomException:
    def __init__(self,message="Error Raised", errors=None):
        super().__init__(message,errors)