import json
from session_memory import session_memory
from config import client, uip_id
from utils import logger
from user_intent_profile.models import UserIntentProfile
from user_intent_profile.functions import wait_for_run_completion, process_tool_calls

def run_user_intent_loop():
    profile=UserIntentProfile()
    profile.mark_start()
    thread_id = client.beta.threads.create().id
    first_agent_msg="Hello! I'm here to help you conduct a competitive analysis. Can you tell me about what you're looking for as well as what your role is at Claritas?"
    print(f"Agent:\n{first_agent_msg}")

    last_agent_msg=first_agent_msg
    last_tool_used=None
    updated_fields=[]
    pending_tool_followup=False

    while not profile.system_workflow.is_profile_complete:
        if not pending_tool_followup:
            user_input=input('You:\n').strip()
            if not user_input:
                continue
            if user_input.lower()=='exit':
                break

            clarification_prompt=profile.build_agent_prompt(
                  user_input=user_input
                , last_agent_msg=last_agent_msg
                , last_tool=last_tool_used
                , updated_fields=updated_fields
            )
            client.beta.threads.messages.create(
                  thread_id=thread_id
                , role='user'
                , content=clarification_prompt
            )
            profile.log_conversation_turn(speaker='user', message=user_input)
            updated_fields.clear()
            last_tool_used=None

        run=client.beta.threads.runs.create(
              thread_id=thread_id
            , assistant_id=uip_id
        )
        run_status=wait_for_run_completion(thread_id,run.id)

        if run_status.status=='requires_action':
            while run_status.status=='requires_action':
                tool_calls=run_status.required_action.submit_tool_outputs.tool_calls
                for call in tool_calls:
                    tool_name=call.function.name
                    args_json=call.function.arguments
                    last_tool_used=tool_name
                    args_dict=json.loads(args_json)
                    updated_fields.extend(args_dict.keys())
                process_tool_calls(thread_id, run.id, profile, run_status)
                run_status=wait_for_run_completion(thread_id, run.id)

            profile.post_state_to_thread(thread_id)
            pending_tool_followup=True
            continue
        elif run_status.status=='completed':
            messages=client.beta.threads.messages.list(thread_id=thread_id)
            for msg in messages.data:
                if msg.run_id==run.id and msg.role=='assistant':
                    last_agent_msg=msg.content[0].text.value
                    print(f'Agent:\n{last_agent_msg}')
                    profile.log_conversation_turn(speaker='agent',message=last_agent_msg)
                    pending_tool_followup=False
                    break
        else:
            raise RuntimeError(f"Unexpected run status: '{run_status.status}'")
    print("\n[Complete] User intent clarification finalized.\n")
    session_memory.save_user_intent_profile(profile)
    return profile