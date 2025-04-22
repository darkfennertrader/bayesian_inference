def debug_shape(name, tensor):
    shape = getattr(tensor, "shape", "<no shape>")
    device = getattr(tensor, "device", "<no device>")
    print(f"\n[DEBUG] {name}.shape={shape}, device={device}")
