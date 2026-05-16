import traceback

def converttime(timeinseconds):
    hours = int(timeinseconds // 3600)
    minutes = int((timeinseconds - 3600 * hours) // 60)
    seconds = int((timeinseconds - 3600 * hours - 60 * minutes) // 1)
    ms = round((timeinseconds - 3600 * hours - 60 * minutes - seconds) * 1000)
    return hours,minutes,seconds,ms

def get_error_detail(exc: Exception) -> str:
    """Abnormal information extraction"""
    tb_list = traceback.extract_tb(exc.__traceback__)
    if not tb_list:
        return f"Unknown error: {str(exc)}"
    tb = tb_list[-1]  
    lineno = tb.lineno
    funcname = tb.name
    exc_msg = str(exc)
    return f"Line {lineno} in {funcname}, Error: {exc_msg}"