import json
import time
from config import client
from utils import logger
from user_intent_profile.models import UserIntentProfile, ResearchFocus, TargetCompany

def get_run_status(thread_id: str, run_id: str):
    return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

def wait_for_run_completion(thread_id: str, run_id: str, time_interval: float=0.5):
    while True:
        run_status=get_run_status(thread_id, run_id)
        if run_status.status in ['completed', 'failed', 'cancelled', 'expired', 'requires_action']:
            return run_status
        time.sleep(time_interval)

def apply_tool_call_to_profile(profile: UserIntentProfile, tool_name: str, args_json: str) -> None:
    args = json.loads(args_json)

    if tool_name=='set_is_profile_complete':
        if args['is_profile_complete']==True:
            profile.log_tool_call(tool_name=tool_name, args=str(args['is_profile_complete']))
            profile.mark_end()
    elif tool_name=='set_corporate_function':
        profile.log_tool_call(tool_name=tool_name, args=str(args['corporate_function']))
        profile.customer_profile.corporate_function=args['corporate_function']
    elif tool_name=='set_product_area':
        profile.log_tool_call(tool_name=tool_name, args=str(args['product_area']))
        profile.customer_profile.product_area=args['product_area']
    elif tool_name=='set_job_focus':
        profile.log_tool_call(tool_name=tool_name, args=str(args['job_focus']))
        profile.customer_profile.job_focus=args['job_focus']
    elif tool_name=='create_research_focus':
        profile.log_tool_call(tool_name=tool_name, args=str(args['new_research_focus_objects']))
        count=args['new_research_focus_objects']
        for _ in range(count):
            profile.research_focus.append(ResearchFocus())
    elif tool_name in ['set_target_companies','set_target_market','set_target_capabilities', 'set_temporal_scope', 'set_desired_outputs', 'set_business_use_case']:
        rf_id=args.get('research_focus_id','').strip()
        if not rf_id:
            logger.error("Assistant didn't return rf_id")
        match=profile.get_rf_id(rf_id)
        if not match:
            logger.error(f"ResearchFocus object '{rf_id}' not found")
        if tool_name=='set_target_companies':
            profile.log_tool_call(tool_name=tool_name, args=str(args))
            company=TargetCompany(
                  name=args['name']
                , source=args['source']
                , seed_for_expansion=args['seed_for_expansion']
            )
            match.target_companies = match.target_companies or []
            match.target_companies.append(company)
        elif tool_name=='set_target_market':
            profile.log_tool_call(tool_name=tool_name, args=str(args['target_market']))
            match.target_market=args['target_market']
        elif tool_name=='set_target_capabilities':
            profile.log_tool_call(tool_name=tool_name, args=str(args['target_capabilities']))
            match.target_capabilities=args['target_capabilities']
        elif tool_name=='set_temporal_scope':
            profile.log_tool_call(tool_name=tool_name, args=str(args['temporal_scope']))
            match.temporal_scope=args['temporal_scope']
        elif tool_name=='set_desired_outputs':
            profile.log_tool_call(tool_name=tool_name, args=str(args['desired_outputs']))
            match.desired_outputs=args['desired_outputs']
        elif tool_name=='set_business_use_case':
            profile.log_tool_call(tool_name=tool_name, args=str(args['business_use_case']))
            match.business_use_case=args['business_use_case']

def process_tool_calls(thread_id: str, run_id: str, profile: UserIntentProfile, run_status):
    tool_calls=run_status.required_action.submit_tool_outputs.tool_calls
    tool_outputs=[]
    for call in tool_calls:
        tool_name=call.function.name
        args_json=call.function.arguments
        call_id=call.id
        apply_tool_call_to_profile(profile, tool_name, args_json)
        tool_outputs.append({
              'tool_call_id':call_id
            , 'output':'OK'
        })
    if tool_outputs:
        client.beta.threads.runs.submit_tool_outputs(
              thread_id=thread_id
            , run_id=run_id
            , tool_outputs=tool_outputs
        )