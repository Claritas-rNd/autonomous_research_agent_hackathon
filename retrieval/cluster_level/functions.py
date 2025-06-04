from retrieval.record_level.functions import record_level_rag
from storage.knowledge_base import KnowledgeBase
from session_memory import session_memory

from typing import List

from utils import logger, embed_text, cosine_similarity, batch_items
from config import client

import json

def cluster_level_retrieval(records: List[str]):
    profile_query=session_memory.load_profile_query()
    profile_embedding=embed_text(profile_query)
    kb=KnowledgeBase().load_all()
    filtered_clusters=[]
    total_records=len(kb)
    if total_records < 20:
        threshold = 0.35
    elif total_records < 50:
        threshold = 0.45
    elif total_records < 100:
        threshold = 0.55
    else:
        threshold = 0.6
    for record in kb:
        if record.record_id not in records:
            continue
        if not record.paragraph_clusters:
            continue
        for cluster in record.paragraph_clusters:
            if not cluster.extracted_facts:
                continue
            sim=cosine_similarity(profile_embedding, cluster.embedding)
            if sim >= threshold:
                filtered_clusters.append({
                      'cluster_id':cluster.cluster_id
                    , 'record_id':cluster.record_id
                    , 'similarity':round(sim, 3)
                    , 'text':cluster.text
                    , 'source_url':record.url
                    , 'source_title':record.title
                })
    return filtered_clusters

def cluster_level_rag(selected_ids: List[str]):
    profile=session_memory.load_user_intent_profile()
    filtered_clusters=cluster_level_retrieval(selected_ids)
    selected_clusters=[]
    for batch in batch_items(filtered_clusters):
        try:
            response=client.responses.create(
                  model='gpt-4.1-mini'
                , input=json.dumps({'user_intent_profile':profile.model_dump(exclude={'metadata'}), 'records': batch}, indent=2)
                , instructions="""
                    You are assisting with the second stage of a Retrieval-Augmented Generation (RAG) process.
                    Your task is to review detailed paragraph-level clusters from pre-selected records and decide which specific clusters should be included as source material for the final response.
                    You are given:
                        - A structured `UserIntentProfile` representing the user's clarified research goal.
                        - A list of paragraph clusters. Each cluster includes a topic summary, extracted facts, named entities, and its cosine similarity to the user's intent.
                    Select all clusters that contain content highly relevant to the user's intent. Do not limit your selection arbitrarily—include any cluster that may help generate a strong, informed response.
                    Return only the `cluster_id` values of the clusters you select. You may select as many OR as few of the clusters from as many records as you feel sufficient. 
                    """
                , text={
                    "format":{
                        "type":"json_schema",
                        "name":"cluster_level_rag",
                        "schema":{
                            "type":"object",
                            "properties":{
                                "selected_cluster_ids":{
                                    "type":"array",
                                    "items":{"type":"string"},
                                    "description":"The unique IDs of paragraph clusters selected for final answer generation based on their relevance to the user's clarified intent."
                                }
                            },
                            "required":['selected_cluster_ids'],
                            "additionalProperties":False
                        },
                        "strict":True
                    }
                }
            )
            result=json.loads(response.output[0].content[0].text)
            selected_clusters.extend(result.get('selected_cluster_ids', []))
        except Exception as e:
            logger.error(f"Failed to get cluster level RAG: {e}")
    cluster_ids=list(set(selected_clusters))
    record_count=len(selected_ids)
    cluster_count=len(cluster_ids)
    logger.info(f'{cluster_count} clusters retrieved from {record_count} records.')
    session_memory.save_selected_clusters(cluster_ids)
    return None