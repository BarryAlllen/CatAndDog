import time
from datetime import datetime

time_begin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(time_begin)
time.sleep(3)
time_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(time_end)
print(" ")
print(format(time_end - time_begin))