import logging
from pathlib import Path
import src

# Set up logging
logger = logging.getLogger("scripts.ex")
# logger = logging.getLogger("integrating_knowledge_graphs_with_logic_tensor_netowrks.process_flattening")

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')


project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
logger.info(f"Data directory: {data_dir}")
