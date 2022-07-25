import os

print(f"rank_id: {os.environ['RANK_ID']}")
print(f"device_id: {os.environ['DEVICE_ID']}")
print(f"rank_size: {os.environ['RANK_SIZE']}")
print(f"MA_CURRENT_HOST_IP: {os.environ['MA_CURRENT_HOST_IP']}")

import time
time.sleep(1)
print(os.environ['RANK_ID'])
