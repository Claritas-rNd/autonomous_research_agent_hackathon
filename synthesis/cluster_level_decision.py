from config import client
from utils import logger
from storage.knowledge_base import KnowledgeBase
from session_memory import session_memory

from typing import Optional
from collections import defaultdict

import json

def get_cluster_level_resolution() -> Optional[str]:
    selected_clusters=session_memory.load_selected_clusters()
    kb=KnowledgeBase().load_all()
    rows=[]
    for record in kb:
        if not record.paragraph_clusters:
            continue
        for cluster in record.paragraph_clusters:
            if cluster.cluster_id in selected_clusters:
                rows.append({
                      'record_id':record.record_id
                    , 'record_title':record.title
                    , 'published_date':record.published_date
                    , 'source_url':record.url
                    , 'topic_digest':record.topic_digest.summary
                    , 'cluster_id':cluster.cluster_id
                    , 'cluster_text':cluster.text
                })
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['record_id']].append(row)
    output=[]
    for record_id, clusters in grouped.items():
        record=clusters[0]
        output.append(f"record_title: {record['record_title']}")
        output.append(f"published_date: {record['published_date'] or 'Unknown'}")
        output.append(f"source_url: {record['source_url']}")
        output.append(f"record_summary: {record['topic_digest']}")
        output.append(f"paragraph_clusters:")
        for c in clusters:
            output.append(f"              - cluster_text: {c['cluster_text']}")
    cluster_resolution_string='\n'.join(output)
    return cluster_resolution_string

def get_cluster_level_decision():
    profile=session_memory.load_user_intent_profile()
    resolution=get_cluster_level_resolution()
    try:
        response=client.responses.create(
              model='gpt-4.1-mini'
            , input=json.dumps({'user_intent_profile':profile.model_dump(exclude={'metadata'}, serialize_as_any=True), 'records':resolution}, indent=2)
            , instructions="""
                You are an intelligent assistant tasked with determining whether the clusters listed below contain enough content to fulfill the user's research goal.
                Review the `user_intent_profile` and available clusters, and choose one of the following options:
                    1. Fallback to a web search — if the clusters do not provide sufficient detail on all key entities, comparisons, or outputs requested.
                    OR
                    2. Move forward with analysis — only if the clusters fully support the desired insight, covering all required companies or topics.
                If the user is seeking a comparison or evaluation between two or more companies, all must be adequately represented. 
                
                This is extremely important. 
                If any of the target_companies are not present in the retrieval, you CANNOT proceed to answer generation and MUST fallback to a web search. No exceptions.
                
                If any are missing or lack substantive coverage, you must fallback.
                Return:
                    - Your decision (`fallback_to_web_search` True or False)
                    - A concise rationale explaining why this choice was made.
                        - For example: "The knowlege base has plenty of information about 'X', but not much about 'Y'"
                If falling back, clearly state what is missing from the knowledge base so that the system can search more effectively.
                """
            , text={
                "format":{
                    "type":"json_schema",
                    "name":"cluster_level_resolution",
                    "schema":{
                        "type":"object",
                        "properties":{
                            "fallback_to_web_search":{
                                "type":"boolean",
                                "description":"Return True if agent should fallback to web_search"
                            },
                            "rationale":{
                                "type":"string",
                                "description":"Single-sentence rationale on why you made the decision you did."
                            }
                        },
                        "required":["fallback_to_web_search","rationale"],
                        "additionalProperties":False
                    },
                    "strict":True
                }
            }
        )
        decision=json.loads(response.output[0].content[0].text)
        if decision.get('fallback_to_web_search'):
            session_memory.save_fallback_rationale(rationale=decision.get('rationale'))
        return decision
    except Exception as e:
        logger.error(f"Failed to make cluster level decision: {e}")
        return None