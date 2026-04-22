"""
CityAgent – represents a city-level bureau (市局).

Responsibilities:
  - Review and approve/return enterprise submissions within its region
  - Forward approved submissions to ProvinceAgent
  - View enterprises registered under its region
"""
from __future__ import annotations
from datetime import datetime
from .base_agent import BaseAgent, Message


class CityAgent(BaseAgent):
    """Agent acting on behalf of a city-level labour bureau."""

    def _register_handlers(self):
        self.register('list_pending_submissions', self._handle_list_pending)
        self.register('approve_submission', self._handle_approve)
        self.register('return_submission', self._handle_return)
        self.register('list_enterprises', self._handle_list_enterprises)
        self.register('query_enterprise', self._handle_query_enterprise)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_list_pending(self, msg: Message) -> Message:
        p = msg.payload
        region_code = p.get('region_code', '')
        rows = self.db.execute(
            """SELECT s.id, e.name AS enterprise_name, e.org_code,
                      sp.period_key, s.survey_employees, s.status, s.submitted_at
               FROM submissions s
               JOIN enterprises e ON e.id = s.enterprise_id
               JOIN survey_periods sp ON sp.id = s.period_id
               WHERE e.region_code LIKE ? AND s.status='submitted'
               ORDER BY s.submitted_at""",
            (region_code + '%',),
        ).fetchall()
        return self._reply(msg, 'pending_list', {'submissions': [dict(r) for r in rows]})

    def _handle_approve(self, msg: Message) -> Message:
        p = msg.payload
        sub_id = p.get('submission_id')
        user_id = p.get('user_id')
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE submissions
               SET status='city_approved', city_reviewed_by=?, city_reviewed_at=?, updated_at=?
               WHERE id=? AND status='submitted'""",
            (user_id, now, now, sub_id),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'submission_approved', {'submission_id': sub_id, 'level': 'city'})

    def _handle_return(self, msg: Message) -> Message:
        p = msg.payload
        sub_id = p.get('submission_id')
        user_id = p.get('user_id')
        reason = p.get('return_reason', '')
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE submissions
               SET status='returned', city_reviewed_by=?, city_reviewed_at=?,
                   return_reason=?, updated_at=?
               WHERE id=? AND status='submitted'""",
            (user_id, now, reason, now, sub_id),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'submission_returned', {'submission_id': sub_id, 'return_reason': reason})

    def _handle_list_enterprises(self, msg: Message) -> Message:
        p = msg.payload
        region_code = p.get('region_code', '')
        rows = self.db.execute(
            """SELECT id, org_code, name, nature, industry, status, created_at
               FROM enterprises WHERE region_code LIKE ? ORDER BY created_at""",
            (region_code + '%',),
        ).fetchall()
        return self._reply(msg, 'enterprise_list', {'enterprises': [dict(r) for r in rows]})

    def _handle_query_enterprise(self, msg: Message) -> Message:
        p = msg.payload
        eid = p.get('enterprise_id')
        row = self.db.execute(
            "SELECT * FROM enterprises WHERE id=?", (eid,)
        ).fetchone()
        if row is None:
            return self._reply(msg, 'error', {'error': 'Enterprise not found'})
        return self._reply(msg, 'enterprise_detail', {'enterprise': dict(row)})
