import os.path
from subprocess import Popen, PIPE

class PyMol:
    def __init__(self, path_to_binaries: str, mode:int = 1):
        # Enable gui...
        pymol_path = os.path.join(path_to_binaries, "pymol")
        cmd = pymol_path + ' -pQe'

        # Not full Screen
        if mode != 1: cmd = pymol_path + ' -pQ'

        # Create a subprocess...
        self.pymol = Popen(cmd, shell=True, stdin=PIPE)

    def __del__(self):
        # Turn off stdin...
        self.pymol.stdin.close()

        # Wait until program is over...
        self.pymol.wait()

        # Terminate the subprcess...
        self.pymol.terminate()

    def __call__(self, s):
        # Keep reading the new pymol command as a byte string...
        self.pymol.stdin.write( bytes( (s + '\n').encode() ) )

        # Flush it asap...
        self.pymol.stdin.flush()