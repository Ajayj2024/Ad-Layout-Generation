import sys
from logger import logging


def error_msg_details(error) -> str:
    """Returns error desciption with file_name, line number and the error"""
    _, _, exc_traceback = sys.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename
    line_number = exc_traceback.tb_lineno
    err_desc = f"Error has occured in python script [{file_name}] line numeber [{line_number}] error message [{str(error)}]"

    return err_desc


class CustomeException(Exception):
    def __init__(self, error_msg) -> None:
        super().__init__(error_msg)
        self.error_msg = error_msg_details(error=error_msg)

    def __str__(self) -> str:
        return self.error_msg
