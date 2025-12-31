import ttnn
import torch

try:
    import tt_metal

    print("tt_metal imported")
    print(dir(tt_metal))
    if hasattr(tt_metal, "detail"):
        print("tt_metal.detail found")
        print(dir(tt_metal.detail))
except ImportError:
    print("tt_metal not found")

try:
    print(dir(ttnn))
    if hasattr(ttnn, "device"):
        print("ttnn.device found")
        # Initialize a device to check methods
        # device = ttnn.open_device(0)
        # print(dir(device))
        # ttnn.close_device(device)
except Exception as e:
    print(e)
