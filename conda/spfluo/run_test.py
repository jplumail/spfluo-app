import spfluo
import sys

if hasattr(spfluo, "has_torch"):
    sys.exit(0)
else:
    sys.exit(1)