from enum import IntEnum

class ErrorLogLevel(IntEnum):
    NONE = 0    # Do not write any error log
    BASIC = 1   # Write line number and error message only
    DETAIL = 2  # Write BASIC info + record content that caused the error