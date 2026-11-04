"""So that `python -m vxentarchy` reaches the command line tool."""
import sys

from .cli import main

sys.exit(main())
