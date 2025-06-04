from utils import logger
from storage.models import KnowledgeBaseRecord
from storage.knowledge_base import KnowledgeBase
from web_search.functions import get_approved_domains, run_web_search, build_kb_record
from urllib.parse import urlparse
from typing import List
from playwright.async_api import async_playwright
import asyncio
from session_memory import session_memory

async def perform_web_search():
    logger.info('Starting web search pipeline...')
    kb = KnowledgeBase()
    knowledge_base_records: List[KnowledgeBaseRecord] = []
    api_results = run_web_search()
    logger.info(f"Found {len(api_results.get('organic_results', []))} results.")
    domain_list = get_approved_domains()
    approved_domains = set(domain_list + ['youtube.com', 'youtu.be'])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        html_tasks = []
        for result in api_results.get('organic_results', []):
            url = result.get('link')
            url_domain = urlparse(url).netloc
            if not any(url_domain == d or url_domain.endswith(f".{d}") for d in approved_domains):
                logger.warning(f"Skipping: '{url_domain}' not in approved domain list.")
                continue
            if kb.contains_url(url):
                logger.warning(f"Skipping: '{url}' already exists in knowledge base.")
                continue
            record = build_kb_record(result)
            if not record:
                logger.error(f"Failed to build record for url: '{url}'")
                continue
            if record.source_type == 'html':
                html_tasks.append((record, browser))
                knowledge_base_records.append(record)
            elif record.source_type == 'pdf':
                record.run_pdf_extraction()
                knowledge_base_records.append(record)
            else:
                logger.warning(f"Skipping: '{url}' is an unsupported source type: {record.source_type}")

        await asyncio.gather(*(r.run_html_extraction(browser) for r, browser in html_tasks))

    if knowledge_base_records:
        kb.save_records(knowledge_base_records)
        session_memory.save_session_records(record_ids=[r.record_id for r in knowledge_base_records])
        logger.info(f"Saved {len(knowledge_base_records)} new record(s) to knowledge base.")
    else:
        logger.info("No new records to save.")