import json
from pathlib import Path
from typing import Iterator, List
from storage.models import KnowledgeBaseRecord

class KnowledgeBase:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.path = project_root / 'storage' / 'knowledge_base.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def save_records(self, records: list[KnowledgeBaseRecord]) -> None:
        existing = self.load_all()
        combined = existing + records
        with self.path.open('w', encoding='utf-8') as f:
            json.dump([r.model_dump() for r in combined], f, ensure_ascii=False, indent=2)

    def load_all(self) -> list[KnowledgeBaseRecord]:
        if self.path.stat().st_size == 0:
            return []
        with self.path.open('r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return [KnowledgeBaseRecord(**r) for r in data]
            except json.JSONDecodeError:
                return []

    def iter_records(self) -> Iterator[KnowledgeBaseRecord]:
        if self.path.stat().st_size == 0:
            return iter([])
        with self.path.open('r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return (KnowledgeBaseRecord(**r) for r in data)
            except json.JSONDecodeError:
                return iter([])

    def contains_url(self, url: str) -> bool:
        for record in self.iter_records():
            if record.url == url:
                return True
        return False

    def get_by_url(self, url: str) -> KnowledgeBaseRecord | None:
        for record in self.iter_records():
            if record.url == url:
                return record
        return None

    def get_by_record_ids(self, record_ids: List[str]) -> List[KnowledgeBaseRecord]:
        return [record for record in self.iter_records() if record.record_id in record_ids]

    def delete_by_url(self, url: str) -> None:
        all_records = [r for r in self.load_all() if r.url != url]
        self.overwrite_all(all_records)

    def overwrite_all(self, records: list[KnowledgeBaseRecord]) -> None:
        with self.path.open('w', encoding='utf-8') as f:
            json.dump([r.model_dump() for r in records], f, ensure_ascii=False, indent=2)

    def save_if_new(self, record: KnowledgeBaseRecord) -> bool:
        if not self.contains_url(record.url):
            self.save_records([record])
            return True
        return False
