from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from enum import Enum
from uuid import uuid4
from datetime import datetime

from utils import logger

class SystemWorkflow(BaseModel):
    is_profile_complete: bool=Field(default=False, description="A status indicator for whether the User Intent Profile is fully populated. Set to true only when all required fields contain specific, unambiguous values that together provide a complete picture of the user’s identity, their goal, the entities involved, and the type of information they need. If any required field is missing or too vague to interpret, this should remain false.")

CorporateFunctionType = Literal["Product", "Marketing", "Sales", "Technology", "Data Science"]
DesiredOutputType = Literal[
    "one_sentence_answer",
    "paragraph_summary",
    "multi_paragraph_summary",
    "sectioned_report",
    "comprehensive_document"
]

class CustomerProfile(BaseModel):
    corporate_function: Optional[CorporateFunctionType]=Field(default=None, description="The user’s primary business function within their organization. This is a fixed, high-level classification that reflects how the user contributes to the company’s objectives. It must be one of the following: Product, Marketing, Sales, Technology, or Data Science. This field does not describe the user’s team, project, or workstream—it only captures their overarching corporate role.")
    product_area: Optional[str]=Field(default=None, description="The specific product, team, or operational area the user is aligned to within their corporate function. This field provides fine-grained context—such as Identity Graph, Audience Creation, Campaign Measurement, or Retail Media—and is used to tailor the research intent to the user’s domain. The value should reflect how the user frames their work internally and should be specific, not generic.")
    job_focus: Optional[List[str]]=Field(default=None, description="Describes the specific goals, responsibilities, or workstreams the user is focused on that led to this request. Each item should capture a distinct purpose or objective tied to the user’s job (e.g., “create competitive slides”, “prepare for sales pitch”, “compare partner offerings”). These values help explain what the user is trying to achieve through the research.")

class ProfileMetadata(BaseModel):
    conversation_turns: List[Dict[str, str]]=Field(default_factory=list)
    tool_calls: List[Dict[str, str]]=Field(default_factory=list)
    start_time: Optional[str]=None
    end_time: Optional[str]=None
    run_time_seconds: Optional[int]=None

class TargetCompany(BaseModel):
    name: str=Field(description="The name of the company relevant to this research focus.")
    source: Literal["user_provided", "agent_generated", "imported", "Admin"]=Field(description="How this company was identified for inclusion—provided directly by the user, inferred by the agent, or sourced from external input.")
    seed_for_expansion: bool=Field(description="Whether this company served as the initial seed for generating a broader competitive set.")

class ResearchFocus(BaseModel):
    research_focus_id: str=Field(default_factory=lambda: str(uuid4()))
    target_companies: Optional[List[TargetCompany]]=Field(description="A structured list of companies relevant to this research focus. Each entry includes metadata indicating how and why the company was included—whether it was explicitly named by the user, inferred by the agent, or added through competitive set expansion. This field allows the agent to reason about the origin and role of each company during research, comparison, and synthesis.", default=None)
    target_market: Optional[str]=Field(description="The product market in which the research is focused. In MarTech and AdTech, this refers to a set of products or solutions that serve a common business need—such as measurement, targeting, or identity resolution—for customers including brands, agencies, retailers, publishers, and data or media companies. A market defines the context in which different companies operate and compete to solve the same type of customer problem.", default=None)
    target_capabilities: Optional[List[str]]=Field(description="A list of specific functional capabilities that the user wants to investigate or compare across companies. Each item should represent a discrete solution function that satisfies part of the broader market need — for example, a method of measurement, targeting, optimization, or data integration. These capabilities must be concrete enough to evaluate individually but general enough to appear across multiple vendors operating in the same market.", default_factory=list)
    temporal_scope: Optional[str]=Field(description="A description of the time period the user wants to focus on for the research. This may include exact dates (e.g., “2023”) or relative periods (e.g., “last 6 months”). It sets boundaries on what information is considered relevant based on when it was created or became true.",default=None)
    desired_outputs: Optional[List[DesiredOutputType]]=Field(description="Specifies the type of output the user expects as a deliverable. Each option represents a distinct level of depth or structure for how the response should be framed—from a one-sentence answer to a comprehensive document. This guides the agent in shaping its response to match the user's expectations for detail, clarity, and completeness. Only one value should be selected per Research Focus to avoid ambiguity in output generation.",default=None)
    business_use_case: Optional[List[str]]=Field(description="A list of concrete business applications for the research output. Each entry should describe how the user plans to use the findings to support a decision, persuade a stakeholder, improve a workflow, or develop internal or external materials. This field captures the downstream purpose of the research and helps the system align the response with the user's intended action or deliverable. Inputs should reflect real-world usage scenarios, not general interests or curiosities.", default_factory=list)

class UserIntentProfile(BaseModel):
    system_workflow: Optional[SystemWorkflow]=Field(description="Contains fields that determine whether the User Intent Profile has enough information to be considered complete.",default_factory=SystemWorkflow)
    customer_profile: Optional[CustomerProfile]=Field(description="Describes the user's role inside their company and the responsibilities influencing their research request.",default_factory=CustomerProfile)
    research_focus: Optional[List[ResearchFocus]]=Field(description="Describes what the user wants to learn and why. This is the part of the profile used to drive search, filtering, and the structure of the answer. There can be multiple research focus entries in one profile.",default_factory=list)
    metadata: Optional[ProfileMetadata]=Field(default_factory=ProfileMetadata)

    def mark_start(self) -> None:
        self.metadata.start_time=datetime.now().isoformat(timespec='seconds')

    def mark_end(self) -> None:
        now=datetime.now()
        self.system_workflow.is_profile_complete=True
        self.metadata.end_time=now.isoformat(timespec='seconds')
        if self.metadata.start_time:
            start=datetime.fromisoformat(self.metadata.start_time)
            self.metadata.run_time_seconds=int((now-start).total_seconds())

    def log_conversation_turn(self, speaker: str, message: str) -> None:
        self.metadata.conversation_turns.append({
              'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M')
            , 'speaker':speaker
            , 'message':message
        })

    def log_tool_call(self, tool_name: str, args: str) -> None:
        logger.info(f'\nTool Call: {tool_name}\nArgument(s): {args}')
        self.metadata.tool_calls.append({
              'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M')
            , 'tool_name':tool_name
            , 'arguments':args
        })

    def get_rf_id(self, rf_id: str) -> Optional[ResearchFocus]:
        return next((rf for rf in self.research_focus if rf.research_focus_id==rf_id), None)

    def get_agent_visible_state(self) -> str:
        lines=[]
        lines.append(f"Is Profile Complete?: {self.system_workflow.is_profile_complete}")
        lines.append(f"Corporate Function: {self.customer_profile.corporate_function}")
        lines.append(f"Product Area: {self.customer_profile.product_area}")
        lines.append(f"Job Focus: {self.customer_profile.job_focus}")
        for rf in self.research_focus:
            lines.append(f"--- Research Focus ID: {rf.research_focus_id} --")
            lines.append(f"Target Companies: {rf.target_companies}")
            lines.append(f"Target Market: {rf.target_market}")
            lines.append(f"Target Capabilities: {rf.target_capabilities}")
            lines.append(f"Temporal Scope: {rf.temporal_scope}")
            lines.append(f"Desired Outputs: {rf.desired_outputs}")
            lines.append(f"Business Use Case: {rf.business_use_case}")
        return '\n'.join(lines)

    def build_agent_prompt(self, user_input: str, last_agent_msg: str, last_tool: str, updated_fields: list[str]) -> str:
        return f"""\
        User Input: {user_input}

        Last Question Asked: {last_agent_msg or "None"}
        Last Tool Used: {last_tool or "None"}
        Updated Fields: {', '.join(updated_fields) if updated_fields else "None"}

        Current Profile State:
        {self.get_agent_visible_state()}

        Instruction: Based on the current profile, decide what information still needs clarification. Avoid repeating resolved fields. Ask only one focused question per turn.
        """

    def post_state_to_thread(self, thread_id: str) -> None:
        from config import client
        content=f"You just made a tool call. Here is the updated profile state:\n\n{self.get_agent_visible_state()}"
        client.beta.threads.messages.create(
              thread_id=thread_id
            , role='user'
            , content=content
        )