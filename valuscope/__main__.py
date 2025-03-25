"""
Main entry point for the ValuScope package.
This allows running the package with 'python -m valuscope'.
"""

import sys
import logging
from valuscope.main import main

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("valuscope_analysis.log"),
            logging.StreamHandler(),
        ],
    )

    # Run the main function with command line arguments
    sys.exit(main())
