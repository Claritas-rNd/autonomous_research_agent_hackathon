from config import client
from utils import logger, embed_text, cosine_similarity, batch_items
from typing import List, Dict, Any, Generator, Optional

from storage.knowledge_base import KnowledgeBase
from session_memory import session_memory

import pandas as pd
import json

def record_level_retrieval() -> Dict:
    profile_query=session_memory.load_profile_query()
    profile_embedding=embed_text(profile_query)
    session_records=session_memory.load_session_records()
    if not session_records:
        session_records=[]
    similarity_rows=[]
    kb=KnowledgeBase().load_all()
    for record in kb:
        if not record.paragraph_clusters:
            continue
        for cluster in record.paragraph_clusters:
            similarity_rows.append({
                  'cluster_id':cluster.cluster_id
                , 'record_id':cluster.record_id
                , 'sim':cosine_similarity(profile_embedding, cluster.embedding)
            })
    similarity_df=pd.DataFrame(similarity_rows)
    grouped_similarity=similarity_df.groupby('record_id').agg(mean_similarity=('sim','mean')).reset_index()
    final_rows=[]
    for record in kb:
        origin='knowledge_base'
        if record.record_id in session_records:
            origin='session_web_search'
        mean_similarity=grouped_similarity[grouped_similarity['record_id']==record.record_id]['mean_similarity']
        similarity_score=float(mean_similarity.values[0]) if not mean_similarity.empty else 0.0
        row={
              'record_id':record.record_id
            , 'title':record.title if record.title else None
            , 'topic':record.topic_digest.topic if record.topic_digest else None
            , 'summary':record.topic_digest.summary if record.topic_digest else None 
            , 'mean_similarity':similarity_score
            , 'source_origin':origin
            , 'source_type':record.source_type
            , 'published_date':record.published_date if record.published_date else 'unknown' 
            , 'url':record.url
            , 'word_count':record.word_count if record.word_count else 'unknown'
        }
        final_rows.append(row)
    return final_rows

def record_level_rag() -> Optional[List[str]]:
    profile=session_memory.load_user_intent_profile()
    record_level_retrieval_records=record_level_retrieval()
    selected_ids=[]
    for batch in batch_items(record_level_retrieval_records):
        try:
            response=client.responses.create(
                  model='gpt-4.1-mini'
                , input=json.dumps({'user_intent_profile': profile.model_dump(exclude={'metadata'}),'records': batch}, indent=2)
                , instructions="""
                    You are an intelligent assistant helping with a RAG process.
                    Your goal is NOT to answer the user's request, but to identify which records should be further explored to eventually answer the user's request.
                    You will recieve a populated `UserIntentProfile` to guide your decision making.
                    Here are the explantions of each field in the records:
                    - record_id: The unique ID of the record. Use this when selecting records.
                    - title: Title of the source
                    - topic: A single word/phrase topic extracted from the source
                    - summary: A short, content-specific summary of what the source is about
                    - mean_similiarity: Cosine similarity score between the record's content and the user's intent (higher=more relevant)
                    - source_origin: 'knowledge_base' means the source existed in the knowledge base prior to this session running. 'session_web_search' means it was extracted during this session.
                    - souce_type: The source type (html, pdf, webinar)
                    - published_date: Date the source was published
                    - url: The URL of the source
                    - word_count: Number of words in the full document
                    If a record appears higly relevant to the `UserIntentProfile`, include it.
                    You may include as many records as necessary.
                    Return only the `record_id` value for each record you select. 
                    """
                , text={
                    "format":{
                        "type":"json_schema",
                        "name":"record_level_rag",
                        "schema":{
                            "type":"object",
                            "properties":{
                                "selected_record_ids":{
                                    "type":"array",
                                    "items":{"type":"string"}
                                }
                            },
                            "required":['selected_record_ids'],
                            "additionalProperties":False
                        },
                        "strict":True
                    }
                }
            )
            result=json.loads(response.output[0].content[0].text)
            selected_ids.extend(result.get('selected_record_ids', []))
        except Exception as e:
            logger.error(f"Failed to get record level RAG: {e}")
    return list(set(selected_ids))