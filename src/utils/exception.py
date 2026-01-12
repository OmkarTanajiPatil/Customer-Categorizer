import sys
from src.utils.logger import logging


def error_message_details(error, error_detail: sys):
    """
    This function extracts detailed information about an exception, including the file name and line number where the error occurred.
    It formats this information into a readable string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"\n\tError occurred in file -> {file_name} \n\t\tLine No -> **{line_number}** \n\t\tError Message -> {str(error)}"
    return error_message


class CustomException(Exception):
    """
    A custom exception class that extends the built-in Exception class.
    It captures detailed error information and logs it using the logger utility.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        """
        It is a method returns error message when exception is raised and printed.
        """
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info(f"{CustomException(e, sys)}")
        print(CustomException(e, sys))
