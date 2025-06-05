from typing import Optional
from pathlib import Path
import io
import time
import json

from synthesis.cluster_level_decision import get_cluster_level_resolution
from config import client, rg_id
from session_memory import session_memory
from utils import logger

def run_response_generation() -> Optional[str]:
    profile = session_memory.load_user_intent_profile()
    corporate_funct = profile.customer_profile.corporate_function.lower()
    output_type=next((output for rf in profile.research_focus if rf.desired_outputs for output in rf.desired_outputs), None)
    customer_path = Path(f"synthesis/customer_profiles/{corporate_funct}.txt")
    output_path=Path(f"synthesis/output_instructions/{output_type}.txt")
    resolution = get_cluster_level_resolution()

    resolution_file = io.BytesIO(resolution.encode('utf-8'))
    resolution_file.name = 'cluster_resolution.txt'

    customer_upload = client.files.create(file=open(customer_path, 'rb'), purpose='assistants')
    resolution_upload = client.files.create(file=resolution_file, purpose='assistants')
    output_upload=client.files.create(file=open(output_path, 'rb'), purpose='assistants')
    
    customer_file_id = customer_upload.id
    resolution_file_id = resolution_upload.id
    output_file_id=output_upload.id
    
    try:
        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role='user',
            content=f"""
                Use the attached files to generate the response.
                Here is the full user intent profile as JSON:
                {json.dumps(profile.model_dump(exclude={'metadata'}, serialize_as_any=True), indent=2)}
                """,
            attachments=[
                {'file_id': customer_file_id, 'tools': [{'type': 'file_search'}]},
                {'file_id': resolution_file_id, 'tools': [{'type': 'file_search'}]},
                {'file_id': output_file_id, 'tools': [{'type': 'file_search'}]}
            ]
        )
        # Start run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=rg_id
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
            if run_status in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(0.5)

        if run_status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            return messages.data[0].content[0].text.value
        else:
            logger.error(f"Failed to generate response: {run_status}")
            return None
    finally:
        for fid in [customer_file_id, resolution_file_id]:
            try:
                client.files.delete(fid)
            except Exception as e:
                logger.error(f"Failed to delete file {fid}: {e}")