import ttnn
import tt_metal

print("ttnn.device attributes:")
print(dir(ttnn.device))
if hasattr(ttnn.device, "WriteToDeviceL1"):
    print("SUCCESS: WriteToDeviceL1 found!")
else:
    print("FAILURE: WriteToDeviceL1 not found.")
