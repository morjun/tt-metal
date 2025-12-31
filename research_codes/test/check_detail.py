try:
    from tt_metal import detail

    print("from tt_metal import detail works")
    print(dir(detail))
except ImportError as e:
    print("ImportError:", e)
    import tt_metal

    if hasattr(tt_metal, "detail"):
        print("tt_metal.detail exists")
        print(dir(tt_metal.detail))
    else:
        print("No detail in tt_metal")
