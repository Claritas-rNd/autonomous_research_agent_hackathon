from web_search.functions import get_content_type
from storage.models import KnowledgeBaseRecord
from utils import logger

import random
import requests
import xml.etree.ElementTree as ET
import asyncio
import aiohttp

from typing import Optional, List, Any, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from tldextract import extract
from datetime import datetime
from aiohttp import ClientSession, ClientTimeout
from pathlib import Path

timeout = ClientTimeout(total=15)

ua_path = Path(__file__).resolve().parent / 'user_agents.txt'
try:
    with open(ua_path, 'r', encoding='utf-8') as f:
        user_agents=[line.strip() for line in f if line.strip()]
except Exception as e:
    user_agents=['Mozilla/5.0']
    logger.error(f'Failed to load user_agents: {e}')

def normalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    if not url.startswith(('http://', 'https://')):
        url='https://'+url
    parsed=urlparse(url)
    netloc=parsed.netloc.lower()
    path=parsed.path.rstrip('/')
    domain_parts=netloc.split('.')
    if len(domain_parts)==2:
        netloc='www.'+netloc
    return f"https://{netloc}{path}"

headers = {
    'User-Agent': random.choice(user_agents),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

def get_sitemap_urls(domain: str) -> List[Dict[str, Any]]:
    start_url=normalize_url(domain)
    parsed=urlparse(start_url)
    base_url=f'{parsed.scheme}://{parsed.netloc}'
    possible_paths=['sitemap.xml', 'sitemap_index.xml']
    collected_links=[]

    def parse_sitemap(content):
        root=ET.fromstring(content)
        for url_tag in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc=url_tag.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            lastmod=url_tag.find("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod")
            changefreq=url_tag.find("{http://www.sitemaps.org/schemas/sitemap/0.9}changefreq")
            priority=url_tag.find("{http://www.sitemaps.org/schemas/sitemap/0.9}priority")
            if loc is not None and loc.text:
                collected_links.append({
                    "url":normalize_url(loc.text.strip()),
                    "lastmod":lastmod.text.strip() if lastmod is not None and lastmod.text else None,
                    "changefreq":changefreq.text.strip() if changefreq is not None and changefreq.text else None,
                    "priority":priority.text.strip() if priority is not None and priority.text else None
                })
    for path in possible_paths:
        try:
            sitemap_url=f'{base_url}/{path}'
            response=requests.get(sitemap_url, headers=headers, timeout=10, allow_redirects=True)
            if response.status_code==200 and 'xml' in response.headers.get("Content-Type","").lower():
                root=ET.fromstring(response.content)
                for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
                    if loc.text:
                        url=normalize_url(loc.text.strip())
                        if url.endswith(".xml"):
                            try:
                                sub_resp=requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                                parse_sitemap(sub_resp.content)
                            except Exception as sub_e:
                                logger.error(f"Failed to get nested sitemap fetch: {url} | {sub_e}") 
                        else:
                            collected_links.append({
                                "url":url,
                                "lastmod":None,
                                "changefreq":None,
                                "priority":None
                            })
                break
        except Exception as e:
            logger.error(f"Failed to fetch sitemap: {sitemap_url} | {e}")
    return collected_links

async def fetch_html(url: str, session: ClientSession, semaphore: asyncio.Semaphore) -> Optional[str]:
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as resp:
                if resp.status==200 and "text/html" in resp.headers.get("Content-Type", "").lower():
                    return await resp.text()
                else:
                    return None
        except Exception as e:
            logger.error(f"Failed to fetch URL: {url} | {e}")
            return None

async def crawl_site(start_url: str, start_lastmod: str, rp: RobotFileParser, sitemap_url_set: set, semaphore: asyncio.Semaphore) -> List:
    downloads_info=[]
    visited=set()
    to_visit=[(start_url, [start_url], start_lastmod)]
    start_domain_parts=extract(start_url)
    start_registered_domain=f"{start_domain_parts.domain}.{start_domain_parts.suffix}"
    download_extensions=('.pdf')
    depth_limit=5

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def process_url(current_url, hierarchy, lastmod):
            if current_url in visited:
                return
            visited.add(current_url)
            if len(hierarchy) > depth_limit:
                return
            html=await fetch_html(current_url, session, semaphore)
            if not html:
                return
            soup=BeautifulSoup(html, 'html.parser')
            for a_tag in soup.find_all('a', href=True):
                raw_href=a_tag['href']
                full_url=urljoin(current_url, raw_href)
                if not full_url.startswith(('https://', 'https://')):
                    continue
                normalized_url=normalize_url(full_url)
                if not normalized_url:
                    continue
                if normalized_url.lower().endswith(download_extensions):
                    if rp is None or rp.can_fetch('*', normalized_url):
                        downloads_info.append({
                            "download_url":normalized_url,
                            "hierarchy":hierarchy.copy(),
                            "lastmod":lastmod
                        })
                    continue
                if rp and not rp.can_fetch('*', normalized_url):
                    continue
                target_domain_parts=extract(normalized_url)
                target_registered_domain=f"{target_domain_parts.domain}.{target_domain_parts.suffix}"
                if target_registered_domain != start_registered_domain:
                    continue
                if urlparse(normalized_url).netloc == urlparse(start_url).netloc:
                    continue
                if normalized_url in visited:
                    continue
                if normalized_url in sitemap_url_set:
                    continue
                to_visit.append((normalized_url, hierarchy+[normalized_url], lastmod))
        while to_visit:
            batch=[]
            while to_visit and len(batch) < 10:
                batch.append(to_visit.pop())
            tasks=[
                process_url(current_url, hierarchy, lastmod) 
                for current_url, hierarchy, lastmod in batch
            ]
            await asyncio.gather(*tasks)
    return downloads_info

def parse_lastmod(lastmod):
    if not lastmod:
        return datetime.min
    try:
        return datetime.fromisoformat(lastmod.replace('Z', '+00:00'))
    except Exception:
        return datetime.min

def deduplicate_downloads(downloads_info):
    deduped={}
    for record in downloads_info:
        key=record['download_url']
        if key not in deduped:
            deduped[key]=record
        else:
            existing=deduped[key]
            existing_lastmod=parse_lastmod(existing["lastmod"])
            new_lastmod=parse_lastmod(record["lastmod"])
            if new_lastmod > existing_lastmod:
                deduped[key]=record
            elif new_lastmod==existing_lastmod:
                if len(record['hierarchy']) < len(existing['hierarchy']):
                    deduped[key]=record
    return list(deduped.values())

def build_kb_record_from_crawl(download: dict) -> Optional[KnowledgeBaseRecord]:
    url=download['download_url']
    lastmod=download.get('lastmod')
    hierarchy=download.get('hierarchy', [])
    published_date=None
    if lastmod and len(hierarchy)==1:
        try:
            published_date=datetime.fromisoformat(lastmod.replace('Z', '+00:00')).date().isoformat()
        except Exception:
            return None
    return KnowledgeBaseRecord(
        url=url,
        url_domain=urlparse(url).netloc,
        title=None,
        source=extract(url).domain,
        source_type=get_content_type(url),
        snippet=None,
        snippet_highlighted_words=None,
        published_date=published_date,
        date_collected=datetime.today().strftime('%Y-%m-%d'),
        last_updated=datetime.today().strftime('%Y-%m-%d'),
        added_by="crawler"
    )