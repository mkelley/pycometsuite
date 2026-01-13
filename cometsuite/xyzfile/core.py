class XYZFileBase:
    """Abstract base class for simulation files."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.__exit__(*args)

    def close(self):
        self._file.close()

    def read(self, b):
        return self._file.read(b)

    def readline(self):
        return self._file.readline()

    def readlines(self, b):
        return self._file.readlines(b)

    def seek(self):
        return self._file.tell()

    def tell(self):
        return self._file.tell()

    def write(self, b):
        return self._file.write(b)

    def writelines(self, b):
        return self._file.writelines(b)
