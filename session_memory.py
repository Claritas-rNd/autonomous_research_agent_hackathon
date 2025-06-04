from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from config import client
from user_intent_profile.models import UserIntentProfile

class SessionMemory(BaseModel):
    user_intent_profile: Optional[UserIntentProfile]=None
    profile_query: Optional[str]=None
    fallback_rationale: List[Dict[str, str | int]] = Field(default_factory=list)
    previous_searches: List[Dict[str, str | int]] = Field(default_factory=list)
    session_records: List[str]=Field(default_factory=list)
    selected_clusters: List[Dict[str, List[str] | int]] = Field(default_factory=list)

    def save_user_intent_profile(self, profile: UserIntentProfile) -> None:
        self.user_intent_profile = profile
    def load_user_intent_profile(self) -> Optional[UserIntentProfile]:
        return self.user_intent_profile if self.user_intent_profile else None

    def save_fallback_rationale(self, rationale: str) -> None:
        self.fallback_rationale.append({
              'fallback_number': len(self.fallback_rationale) + 1
            , 'rationale': rationale
        })
    def load_fallback_rationale(self) -> Optional[List[Dict[str, str | int]]]:
        return self.fallback_rationale if self.fallback_rationale else None

    def save_session_records(self, record_ids: List[str]) -> None:
        for record_id in record_ids:
            self.session_records.append(record_id)

    def load_session_records(self) -> Optional[List[str]]:
        return self.session_records if self.session_records else None 

    def save_previous_searches(self, search_query: str) -> None:
        self.previous_searches.append({
              'search_number': len(self.previous_searches) + 1
            , 'search_query':search_query
        })

    def load_previous_searches(self) -> Optional[List[Dict[str, str | int]]]:
        return self.previous_searches if self.previous_searches else None

    def save_selected_clusters(self, cluster_ids: List[str]) -> None:
        self.selected_clusters.append({
              'RAG_run':len(self.selected_clusters) + 1
            , 'cluster_ids':cluster_ids
        })

    def load_selected_clusters(self) -> Optional[List[str]]:
        if not self.selected_clusters:
            return None
        return self.selected_clusters[-1]['cluster_ids']

    def get_profile_query(self) -> Optional[str]:
        profile = self.load_user_intent_profile()
        if not profile or not profile.research_focus:
            return None
        parts = []
    
        if profile.customer_profile:
            if profile.customer_profile.corporate_function:
                parts.append(f"Corporate Function: {profile.customer_profile.corporate_function}")
            if profile.customer_profile.product_area:
                parts.append(f"Product Area: {profile.customer_profile.product_area}")
            if profile.customer_profile.job_focus:
                parts.append(f"Job Focus(es): {', '.join(profile.customer_profile.job_focus)}")
    
        all_companies = []
        all_markets = []
        all_capabilities = []
    
        for rf in profile.research_focus:
            if rf.target_companies:
                all_companies += [tc.name for tc in rf.target_companies if tc.name]
            if rf.target_market:
                all_markets.append(rf.target_market)
            if rf.target_capabilities:
                all_capabilities += rf.target_capabilities
    
        if all_companies:
            parts.append(f"Target Company(s): {', '.join(sorted(set(all_companies)))}")
        if all_markets:
            parts.append(f"Target Market(s): {', '.join(sorted(set(all_markets)))}")
        if all_capabilities:
            parts.append(f"Target Capabilities: {', '.join(sorted(set(all_capabilities)))}")
    
        return " | ".join(parts)

    def load_profile_query(self) -> str:
        if self.profile_query:
            return self.profile_query
        else:
            self.profile_query=self.get_profile_query()
            return self.profile_query

session_memory=SessionMemory()