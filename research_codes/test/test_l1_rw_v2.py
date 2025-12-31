import ttnn
import tt_metal

# from tt_metal import ttnn as ttnn_metal


def main():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    print("Device opened.")

    # Address to test
    l1_addr = 100000
    test_val = 123456789

    # Grid
    grid = device.compute_with_storage_grid_size()
    print(f"Grid size: {grid.x} x {grid.y}")

    # Pick a core (e.g. 1,1)
    core = ttnn.CoreCoord(1, 1)

    # Write
    print(f"Writing {test_val} to Core {core} at Address {l1_addr}")
    data = [test_val]
    # Use ttnn._ttnn.device.WriteToDeviceL1
    try:
        ttnn._ttnn.device.WriteToDeviceL1(device, core, l1_addr, data)
        print("Write successful (no exception).")
    except Exception as e:
        print(f"Write failed: {e}")
        ttnn.close_device(device)
        return

    # Read back using ReadFromDeviceL1
    # Need to find the API for reading.
    # tt_metal.detail.ReadFromDeviceL1?
    # Or ttnn._ttnn.device.ReadFromDeviceL1?

    # Let's try to verify if ReadFromDeviceL1 is exposed in ttnn._ttnn.device
    # I didn't verify that binding before.
    # But I can use a buffer to read back?

    # Create a buffer at that address?
    # Buffer creation usually allocates. We want to read raw address.

    # Let's try to check ttnn._ttnn.device attributes
    print(dir(ttnn._ttnn.device))

    ttnn.close_device(device)
    print("Device closed.")


if __name__ == "__main__":
    main()
