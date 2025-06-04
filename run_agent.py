from user_intent_profile.user_intent_profile import run_user_intent_loop

from synthesis.record_level_decision import get_record_level_decision
from synthesis.cluster_level_decision import get_cluster_level_decision
from synthesis.answer_generation import run_response_generation

from retrieval.record_level.functions import record_level_rag
from retrieval.cluster_level.functions import cluster_level_rag

from web_search.web_search import perform_web_search

from typing import Optional

import time

from utils import logger

async def run_agent() -> Optional[str]:
    start_time=time.perf_counter()
    profile=run_user_intent_loop()
    record_level_decision=get_record_level_decision()
    if record_level_decision.get('fallback_to_web_search'):
        logger.info(f'Agent has decided to fallback to web search')
        logger.info(f"Rationale: {record_level_decision.get('rationale')}")
        await perform_web_search()
    while True:
        selected_record_ids=record_level_rag()
        cluster_level_rag(selected_record_ids)
        cluster_level_decision=get_cluster_level_decision()
        if not cluster_level_decision.get('fallback_to_web_search'):
            logger.info(f'Agent has decided to proceed to answer generatation')
            logger.info(f"Rationale: {cluster_level_decision.get('rationale')}")
            result=run_response_generation()
            elapsed=time.perf_counter()-start_time
            logger.info(f'Agent response generation completed in {elapsed:.2f} seconds')
            return result
        logger.info(f'Agent has decided to fallback to web_search')
        logger.info(f"Rationale: {cluster_level_decision.get('rationale')}")
        await perform_web_search()