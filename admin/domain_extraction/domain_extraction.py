import asyncio, time

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from typing import List

from utils import logger
from storage.models import KnowledgeBaseRecord
from storage.knowledge_base import KnowledgeBase
from domain_extraction.functions import get_sitemap_urls, crawl_site, deduplicate_downloads, build_kb_record_from_crawl

async def run_domain_extraction(domain: str):
    start_time = time.time()
    logger.info(f'Starting domain extraction for: {domain}')
    all_downloads_info = []

    collected_links = get_sitemap_urls(domain)
    collected_links = list({link_info['url']: link_info for link_info in collected_links}.values())
    global_visited_sitemaps = {link_info['url'] for link_info in collected_links}
    logger.info(f'Collected {len(collected_links)} sitemap URLs from {domain}')

    parsed_start = urlparse(collected_links[0]['url'])
    robots_url = f"{parsed_start.scheme}://{parsed_start.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception as e:
        logger.error(f"Failed to read robots.txt for {parsed_start.netloc}: {e}")
        return

    outer_semaphore = asyncio.Semaphore(7)
    inner_semaphore = asyncio.Semaphore(25)

    async def limited_crawl(link_info):
        async with outer_semaphore:
            return await crawl_site(
                start_url=link_info['url'],
                start_lastmod=link_info.get('lastmod'),
                rp=rp,
                sitemap_url_set=global_visited_sitemaps,
                semaphore=inner_semaphore
            )

    tasks = [limited_crawl(link_info) for link_info in collected_links]
    results = await asyncio.gather(*tasks)
    all_downloads_info = [item for sublist in results for item in sublist]

    elapsed = time.time() - start_time
    logger.info(f'Download link discovery completed in {int(elapsed // 60)}m {int(elapsed % 60)}s')
    logger.info(f'Total downloads before deduplication: {len(all_downloads_info)}')
    all_downloads_info = deduplicate_downloads(all_downloads_info)
    logger.info(f'Total downloads after deduplication: {len(all_downloads_info)}')

    kb = KnowledgeBase()
    knowledge_base_records: List[KnowledgeBaseRecord] = []

    last_log_time = time.time()
    total = len(all_downloads_info)

    for idx, download in enumerate(all_downloads_info):
        url = download.get('download_url', '')
        path = urlparse(url).path.split('/')[-1]
        logger.info(f'Processing [{idx+1}/{total}]: {path}')

        if kb.contains_url(url):
            logger.warning(f"Skipping: '{url}' already exists in knowledge base.")
            continue

        try:
            record = build_kb_record_from_crawl(download)
            if record.source_type != 'pdf':
                logger.info(f'Skipped non-PDF: {path}')
                continue

            record_start = time.time()
            record.run_pdf_extraction()

            if not record.paragraph_clusters:
                logger.warning(f'No content extracted from: {path}')
                continue

            logger.info(f'Finished PDF extraction in {int(time.time() - record_start)}s: {path}')
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            continue

        if not record.title:
            filename = urlparse(record.url).path.split('/')[-1]
            if filename.lower().endswith('.pdf'):
                filename = filename[:-4]
            record.title = filename.replace('-', ' ').replace('_', ' ').strip()

        if not record.snippet:
            all_paragraphs = [cluster.text for cluster in record.paragraph_clusters]
            if all_paragraphs:
                record.snippet = ' '.join(all_paragraphs)[:500]

        knowledge_base_records.append(record)

    if knowledge_base_records:
        kb.save_records(knowledge_base_records)
        logger.info(f'Knowledge base saved.')

    logger.info(f'Completed domain extraction for {domain}: {len(knowledge_base_records)} new records saved.')