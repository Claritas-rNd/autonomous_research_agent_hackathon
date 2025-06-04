from pydantic import BaseModel, Field
from uuid import uuid4
from typing import List, Optional
from utils import embed_text, logger
from bs4 import BeautifulSoup
from io import BytesIO
from playwright.async_api import async_playwright
from pymupdf4llm import to_markdown
from config import client
from datetime import datetime
from urllib.parse import urlparse

import requests
import json
import fitz
import spacy
nlp=spacy.load('en_core_web_sm')

class ExtractedFact(BaseModel):
    fact_id: str=Field(default_factory=lambda: str(uuid4()))
    cluster_id: str
    record_id: str

    entity: str
    claim: str

class ParagraphCluster(BaseModel):
    cluster_id: str=Field(default_factory=lambda: str(uuid4()))
    record_id: str
    
    text: str
    embedding: List[float]

    extracted_facts: Optional[List[ExtractedFact]]=None
    
    def get_extracted_facts(self) -> None:
        try:
            response = client.responses.create(
                  model = 'gpt-4.1-nano'
                , input = self.text
                , instructions = """
                    Extract all verifiable facts from the input text. 
                    For each fact, identify the primary entity it refers to and the specific claim made about that entity. 
                    Return arrays of entities and claims, where each claim corresponds to the entity at the same position in the other array. 
                    Only include facts that are explicit and objectively stated in the text.
                  """
                , text = {
                    "format": {
                          "type": "json_schema"
                        , "name": "extracted_fact"
                        , "schema": {
                              "type": "object"
                            , "properties": {
                                  "entity": {
                                      "type": "array"
                                    , "items": { "type": "string" }
                                    , "description": "The primary entity the text is about."
                                  }
                                , "claim": {
                                      "type": "array"
                                    , "items": { "type": "string" }
                                    , "description": "The verifiable fact from the text about the entity."
                                  }
                              }
                            , "required": ["entity", "claim"]
                            , "additionalProperties": False
                        }
                    }
                }
            )
            parsed = json.loads(response.output[0].content[0].text)
            entity_claims = [{"entity": e, "claim": c} for e, c in zip(parsed["entity"], parsed["claim"])]
            self.extracted_facts = [
                ExtractedFact(
                      cluster_id = self.cluster_id
                    , record_id = self.record_id
                    , entity = item["entity"]
                    , claim = item["claim"]
                ) for item in entity_claims
            ]
        except Exception as e:
            logger.error(f"Failed to get extracted fact: {e}")

class TopicDigest(BaseModel):
    digest_id: str=Field(default_factory=lambda: str(uuid4()))
    record_id: str
    
    topic: str
    summary: str

class KnowledgeBaseRecord(BaseModel):
    record_id: str=Field(default_factory=lambda: str(uuid4()))
    url: str
    url_domain: str
    title: Optional[str]=None
    source: Optional[str]=None
    source_type: Optional[str]=None

    snippet: Optional[str]=None
    snippet_highlighted_words: Optional[List[str]]=None
    named_entities: Optional[List[str]]=None
    word_count: Optional[int]=None
    image_present: Optional[bool]=None

    published_date: Optional[str]=None
    date_collected: Optional[str]=None
    last_updated: Optional[str]=None
    added_by: Optional[str]=None

    paragraph_clusters: Optional[List[ParagraphCluster]]=None
    topic_digest: Optional[TopicDigest]=None

    def get_named_entities(self, main_text: str) -> None:
        try:
            doc=nlp(main_text)
            self.named_entities=list(set(ent.text.strip() for ent in doc.ents if ent.label_ in {"ORG"}))
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
    
    def get_topic_digest(self, full_text: str) -> None:
        try:
            response = client.responses.parse(
                  model = 'gpt-4.1-mini'
                , input = full_text.strip()
                , instructions = """
                    Read the full text and determine the main topic it focuses on. 
                    Choose a single word or short phrase for the topic. 
                    Then, summarize what the text says about that topic in 2–5 clear, factual sentences.
                    Do not include opinions, unrelated content, or vague statements. Keep it focused on the core message.
                  """
                , text = {
                    "format": {
                          "type": "json_schema"
                        , "name": "topic_digest"
                        , "schema": {
                              "type": "object"
                            , "properties": {
                                  "topic": {
                                      "type": "string"
                                    , "description": "A single word or phrase that describes what the text is about"
                                  }
                                , "summary": {
                                      "type": "string"
                                    , "description": "Summary of the topic. 2-5 sentences describing what the source says about the topic."
                                  }
                              }
                            , "required": ["topic", "summary"]
                            , "additionalProperties": False
                        }
                    }
                }
            )
            if not response.output or not response.output[0].content or not response.output[0].content[0].text:
                logger.warning(f"No output from topic digest parse for url: {self.url}")
                return
            parsed = json.loads(response.output[0].content[0].text)
            self.topic_digest = TopicDigest(
                  record_id = self.record_id
                , topic = parsed['topic']
                , summary = parsed['summary']
            )
        except Exception as e:
            logger.error(f"Failed to get topic digest for url ('{self.url}'): {e}")

    async def run_html_extraction(self, browser) -> None:
        try:
            page = await browser.new_page()
            await page.goto(self.url, timeout=30000)
            html = await page.content()
            await page.close()
        except Exception as e:
            logger.error(f"failed to download/parse url ('{self.url}'): {e}")
            return
    
        soup = BeautifulSoup(html, 'html.parser')
        main_text = soup.get_text(separator=' ', strip=True)
        self.word_count = len(main_text.split())
        self.image_present = bool(soup.find_all('img'))
    
        heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        paragraph_clusters: List[ParagraphCluster] = []
        all_elements = soup.find_all(heading_tags + ['p', 'ul', 'ol'])
        current_heading = None
        current_cluster = []
    
        for el in all_elements:
            if el.name in heading_tags:
                if current_heading and current_cluster:
                    heading = current_heading.get_text(strip=True)
                    current_clusters = ' '.join(current_cluster)
                    full_text = f"Heading: {heading}\nText: {current_clusters}"
                    embedding = embed_text(text=f"{heading} - {current_clusters}")
                    cluster = ParagraphCluster(
                          record_id=self.record_id
                        , text=full_text
                        , embedding=embedding
                    )
                    cluster.get_extracted_facts()
                    paragraph_clusters.append(cluster)
                current_heading = el
                current_cluster = []
            elif current_heading:
                if el.name == 'p':
                    para_text = el.get_text(strip=True)
                    if para_text:
                        current_cluster.append(para_text)
                elif el.name in ['ul', 'ol']:
                    items = el.find_all('li')
                    for item in items:
                        item_text = item.get_text(strip=True)
                        if item_text:
                            current_cluster.append(f"-  {item_text}")
    
        if current_heading and current_cluster:
            heading = current_heading.get_text(strip=True)
            current_clusters = ' '.join(current_cluster)
            full_text = f"Heading: {heading}\nText: {current_clusters}"
            embedding = embed_text(text=f"{heading} - {current_clusters}")
            cluster = ParagraphCluster(
                  record_id=self.record_id
                , text=full_text
                , embedding=embedding
            )
            cluster.get_extracted_facts()
            paragraph_clusters.append(cluster)
    
        self.paragraph_clusters = paragraph_clusters
        if main_text:
            self.get_topic_digest(main_text)
            self.get_named_entities(main_text)

    def run_pdf_extraction(self) -> None:
        def is_probable_table_of_contents(text: str) -> bool:
            lowered = text.lower()
            if any(k in lowered for k in ['table of contents', 'contents', 'index']):
                return True
            lines = text.split('\n')
            if len(lines) > 10:
                digit_lines = sum(1 for l in lines if any(char.isdigit() for char in l))
                if digit_lines / len(lines) > 0.6:
                    return True
            if text.count("...") > 10:
                return True
            return False
    
        def resolve_and_download_pdf(url: str) -> Optional[BytesIO]:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            response = requests.get(url, headers=headers, timeout=15)
            if "application/pdf" in response.headers.get("Content-Type", "").lower():
                return BytesIO(response.content)
            soup = BeautifulSoup(response.text, "html.parser")
            embed = soup.find("embed", {"type": "application/pdf"})
            if embed and embed.get("src"):
                resolved_url = embed["src"]
                pdf_resp = requests.get(resolved_url, headers=headers, timeout=15)
                if "application/pdf" in pdf_resp.headers.get("Content-Type", "").lower():
                    return BytesIO(pdf_resp.content)
            return None
    
        pdf_stream = resolve_and_download_pdf(self.url)
        doc = fitz.open(stream=pdf_stream, filetype='pdf')
        md_text = to_markdown(doc)
        paragraph_clusters: List[ParagraphCluster] = []
        current_cluster = []
        self.word_count = len(md_text.split())
        self.image_present = any(page.get_images() for page in doc)
    
        for line in md_text.splitlines():
            if line.strip().startswith("#"):
                if current_cluster:
                    full_text = " ".join(current_cluster).strip()
                    cluster = ParagraphCluster(
                        record_id=self.record_id,
                        text=full_text,
                        embedding=embed_text(full_text)
                    )
                    if self.added_by != 'crawler':
                        cluster.get_extracted_facts()
                    paragraph_clusters.append(cluster)
                current_cluster = [line.strip().lstrip('#').strip()]
            elif line.strip():
                current_cluster.append(line.strip())
        if current_cluster:
            full_text = " ".join(current_cluster).strip()
            cluster = ParagraphCluster(
                record_id=self.record_id,
                text=full_text,
                embedding=embed_text(full_text)
            )
            if self.added_by != 'crawler':
                cluster.get_extracted_facts()
            paragraph_clusters.append(cluster)
    
        paragraph_clusters = [c for c in paragraph_clusters if not is_probable_table_of_contents(c.text)]
        self.paragraph_clusters = paragraph_clusters
        self.get_named_entities(md_text)
        self.get_topic_digest(md_text)