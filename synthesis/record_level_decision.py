import json

from config import client
from utils import logger
from retrieval.record_level.functions import record_level_retrieval
from storage.knowledge_base import KnowledgeBase
from session_memory import session_memory

from typing import Optional

def get_record_level_resolution() -> Optional[str]:
    records=record_level_retrieval()
    rows=[]
    for r in records:
        row=f"""
        [record_id]: {r['record_id']}
        [title]: "{r["title"]}"
        [topic]: {r["topic"]}
        [summary]: {r["summary"]}
        [average similarity]: {round(r['mean_similarity'],3)}
        [source]: {r["source_origin"]}
        [word count]: {r["word_count"]}
        [published]: {r["published_date"]}
        [url]: {r["url"]}
        """
        rows.append(row.strip())
    return '\n\n'.join(rows)

def get_record_level_decision():
    profile=session_memory.load_user_intent_profile()
    resolution=get_record_level_resolution()
    try:
        response=client.responses.create(
              model='gpt-4.1-mini'
            , input=json.dumps({'user_intent_profile':profile.model_dump(exclude={'metadata'}, serialize_as_any=True), "records": resolution}, indent=2)
            , instructions="""
                You are an intelligent assistant tasked with determining whether the records listed below contain enough context to fully answer the user's request.
                Your decision should be based on the `user_intent_profile` and the records provided. You must choose one of the following:
                    1. Fallback to a web search — if any part of the user’s objective cannot be supported with the records available.
                    OR
                    2. Move forward with analysis — only if all required elements are well represented in the records.
                    Pay particular attention to whether the records cover all *target entities or companies* mentioned in the user’s request. 
                    If even one target (e.g., a competitor being compared) is not represented with meaningful content, you should recommend a fallback. There should be NO exceptions.
                Return:
                    - Your decision (`fallback_to_web_search` True or False)
                    - A single-sentence rationale explaining your decision.
                        - For example: "The knowlege base has plenty of information about 'X', but not much about 'Y'"
                If you choose to fallback, the rationale should briefly describe what specific content is missing. This rationale will guide a targeted web search.
                """
            , text={
                "format":{
                    "type":"json_schema",
                    "name":"record_level_decision",
                    "schema":{
                        "type":"object",
                        "properties":{
                            "fallback_to_web_search":{
                                "type":"boolean",
                                "description":"Return True if agent should fallback to web_search"
                            },
                            "rationale":{
                                "type":"string",
                                "description":"Single-sentence rationale on why you made the decision."
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
        logger.error(f"Failed to make record level decision: {e}")
        return None