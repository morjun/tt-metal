import tt_metal
import ttnn

print("tt_metal dir:", dir(tt_metal))
if hasattr(tt_metal, "device"):
    print("tt_metal.device dir:", dir(tt_metal.device))

# Check for Write to L1
candidates = []


def search(obj, path):
    for name in dir(obj):
        if "Write" in name or "write" in name:
            candidates.append(f"{path}.{name}")


search(tt_metal, "tt_metal")
# search(tt_metal.device, "tt_metal.device")
if hasattr(tt_metal, "detail"):
    search(tt_metal.detail, "tt_metal.detail")

print("Candidates:", candidates)
