def converttime(timeinseconds):
    hours = int(timeinseconds // 3600)
    minutes = int((timeinseconds - 3600 * hours) // 60)
    seconds = int((timeinseconds - 3600 * hours - 60 * minutes) // 1)
    ms = round((timeinseconds - 3600 * hours - 60 * minutes - seconds) * 1000)
    return hours,minutes,seconds,ms