import sys

def error_msg_details(error, error_detail:sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    lineno = exc_tb.tb_lineno
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message =  f"Error Occured at Line Number [{lineno}] in the filename [{filename}]. Error is [{error}]"
    return error_message

class Sys_error(Exception):
    def __init__(self, error:str, error_detail:sys) -> None:
        self.error_message = error_msg_details(error, error_detail)
    
    def __str__(self):
        return self.error_message
