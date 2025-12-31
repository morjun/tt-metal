import ttnn

try:
    print("ttnn._ttnn.device attributes:")
    print(dir(ttnn._ttnn.device))
    if hasattr(ttnn._ttnn.device, "WriteToDeviceL1"):
        print("SUCCESS: WriteToDeviceL1 found in ttnn._ttnn.device!")
    else:
        print("FAILURE: WriteToDeviceL1 not found in ttnn._ttnn.device.")
except Exception as e:
    print(e)
