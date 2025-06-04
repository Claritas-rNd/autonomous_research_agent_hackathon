from storage.models import KnowledgeBaseRecord, ParagraphCluster, ExtractedFact, TopicDigest
from session_memory import session_memory
from config import client, SerpAPI_key
from utils import logger
from typing import Optional, Dict
from serpapi import GoogleSearch
from datetime import datetime, time
from urllib.parse import urlparse
from dateutil import parser

import requests
import json

def get_search_query() -> Optional[str]:
    user_intent_profile = session_memory.load_user_intent_profile()
    fallback_rationale = session_memory.load_fallback_rationale() or []
    previous_searches = session_memory.load_previous_searches() or []

    target_companies = []
    target_markets = []
    target_capabilities = []

    if not user_intent_profile:
        return None

    for rf in user_intent_profile.research_focus:
        if rf.target_companies:
            for target_company in rf.target_companies:
                target_companies.append(target_company.name)
        if rf.target_market:
            target_markets.append(rf.target_market)
        if rf.target_capabilities:
            target_capabilities.extend(rf.target_capabilities)

    fallback_str = '\n'.join(f"{r['fallback_number']}. {r['rationale']}" for r in fallback_rationale)
    searches_str = '\n'.join(f"{q['search_number']}. {q['search_query']}" for q in previous_searches)

    profile = f"""
    Target Companies: {', '.join(target_companies)}
    Target Markets: {', '.join(target_markets)}
    Target Capabilities: {', '.join(target_capabilities)}
    
    Fallback Rationale(s):
    {fallback_str}
    
    Previous Search(es):
    {searches_str}
    """
    try:
        response = client.responses.create(
              model='gpt-4.1',
              input=profile,
              instructions="""
                You are a research assistant specializing in crafting high-precision Google search queries.
                You will receive:
                    - A list of target companies
                    - A list of product markets or solution areas
                    - A list of specific functional capabilities to evaluate
                    - A rationale explaining what information is still missing (use this as your primary guidance)
                        - Prioritize the company or content type that is explicitly missing, as described in the rationale. If a company is already well covered, do not include it.
                    - A list of previous search queries already attempted
                Your task:
                    - Generate one short, natural-sounding Google search query focused only on retrieving the missing information described in the rationale.
                    - Prioritize the underrepresented or missing company. Do not include companies already covered.
                    - Emphasize comparative insight, product capabilities, or detailed solution descriptions that align with the user’s research goal.
                    - Use natural phrasing — something a real researcher would type into Google.
                    - If possible, include a site constraint for the missing company's official domain (e.g., site:company.com).
                    - Avoid repeating any previously attempted query.
                    - Do not include years or date ranges unless explicitly instructed.
                    - Do not use filler or broad phrasing — just focus on closing the exact gap in the rationale.
                """
        )
        return response.output[0].content[0].text
    except Exception as e:
        logger.error(f"Failed to generate search query: {e}")
        return None

def run_web_search() -> Dict:
    query=get_search_query()
    params = {
        "q": query,
        "engine": "google",
        "api_key": SerpAPI_key
    }
    search=GoogleSearch(params)
    logger.info(f"Search Query: '{query}'")
    session_memory.save_previous_searches(query)
    return search.get_dict()

def get_approved_domains():
    profile=session_memory.load_user_intent_profile()
    if not profile:
        return None
    target_companies=[]
    for rf in profile.research_focus:
        if not rf.target_companies:
            return None
        for tc in rf.target_companies:
            target_companies.append(tc.name)
    try:
        response = client.responses.create(
              model='gpt-4.1-mini'
            , input=[{
                  'role': 'user',
                  'content': f"Return the official domains for the following companies: {', '.join(target_companies)}"
              }]
            , instructions="""
                You are an expert at identifying official company domains using only verifiable sources. 
                Only return domains you are certain are officially owned and used by the listed companies. 
                Do not guess, infer, or fabricate. If you are not certain, omit the company entirely. 
                Return a list of domains in the format 'company.com'—no subdomains or URLs. 
                Return only verified, primary domains used for official communications.
                """
            , text={
                'format': {
                    'type': 'json_schema',
                    'name': 'domain',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'domains': {
                                'type': 'array',
                                'items': {'type': 'string'}
                            }
                        },
                        'required': ['domains'],
                        'additionalProperties': False
                    },
                    'strict': True
                }
            }
        )
        return json.loads(response.output[0].content[0].text)['domains']
    except Exception as e:
        logger.error(f'Failed to get approved domains: {e}')

def get_content_type(url: str) -> str:
    content_type_mapping = {
        'pdf': 'pdf',
        'html': 'html',
        'video': 'video',
        'json': 'json',
        'xml': 'xml',
        'msword': 'docx',
        'word': 'docx',
        'powerpoint': 'pptx',
        'presentation': 'pptx',
        'excel': 'spreadsheet',
        'spreadsheet': 'spreadsheet'
    }
    if 'youtube.com' in url or 'youtu.be' in url:
        return 'youtube video'
    try:
        response=requests.head(url,allow_redirects=True, timeout=10)
        content_type=response.headers.get('Content-Type','').lower()
        for key, category in content_type_mapping.items():
            if key in content_type:
                return category
        return 'unknown'
    except Exception as e:
        logger.error(f"Failed to get content type for url ('{url}'): {e}")
        return 'error_fetching'

def get_date(input_date) -> Optional[str]:
    try:
        if isinstance(input_date, str):
            parsed = parser.parse(input_date)
        elif isinstance(input_date, datetime):
            parsed = input_date
        elif isinstance(input_date, date):
            parsed = datetime.combine(input_date, datetime.min.time())
        else:
            return None
        return parsed.strftime('%Y-%m-%d')
    except Exception:
        return None

def build_kb_record(result: Dict) -> Optional[KnowledgeBaseRecord]:
    try:
        url = result.get('link')
        url_domain = urlparse(url).netloc
        date_final = None
        if result.get('date'):
            date_final = get_date(input_date=result['date'])

        return KnowledgeBaseRecord(
              url=url
            , url_domain=url_domain
            , title=result.get('title')
            , source=result.get('source')
            , source_type=get_content_type(url=url)
            , snippet=result.get('snippet')
            , snippet_highlighted_words=result.get('snippet_highlighted_words')
            , published_date=date_final
            , date_collected=datetime.today().strftime('%Y-%m-%d')
            , last_updated=datetime.today().strftime('%Y-%m-%d')
            , added_by='web_search_agent'
        )
    except Exception as e:
        logger.error(f"Failed to build record for url: '{result.get('link')}' - {e}")
        return None