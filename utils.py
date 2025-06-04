import logging
from typing import List, Dict, Any, Generator
from config import client
import numpy as np

logging.basicConfig(
      level=logging.INFO
    , format='%(asctime)s - %(levelname)s - %(message)s'
    , datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('utils')

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

def embed_text(text: str) -> List[float]:
    try:
        response = client.embeddings.create(
              model = 'text-embedding-ada-002'
            , input = text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f'Failed to embed text: {e}')

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a=np.array(vec1)
    b=np.array(vec2)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def batch_items(items: List[Dict[str, Any]], batch_size: int=10) -> Generator[List[Any], None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]