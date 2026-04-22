"""
EnterpriseAgent – represents an enterprise user.

Responsibilities:
  - Register / update enterprise filing information
  - Submit monthly employment data
  - Query historical submission status
  - Receive notifications
"""
from __future__ import annotations
from datetime import datetime
from .base_agent import BaseAgent, Message


class EnterpriseAgent(BaseAgent):
    """Agent acting on behalf of an enterprise."""

    def _register_handlers(self):
        self.register('register_enterprise', self._handle_register)
        self.register('update_enterprise', self._handle_update)
        self.register('submit_data', self._handle_submit_data)
        self.register('query_submissions', self._handle_query_submissions)
        self.register('change_password', self._handle_change_password)
        self.register('browse_notifications', self._handle_browse_notifications)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_register(self, msg: Message) -> Message:
        p = msg.payload
        # Validate org_code length
        org_code = p.get('org_code', '').strip()
        if not org_code or len(org_code) > 9:
            return self._reply(msg, 'error', {'error': '组织机构代码不超过9位'})

        now = datetime.now().isoformat()
        try:
            self.db.execute(
                """INSERT INTO enterprises
                   (org_code, name, nature, industry, region_code,
                    contact_person, contact_phone, email, address,
                    main_business, created_by, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    org_code, p.get('name', ''), p.get('nature', ''),
                    p.get('industry', ''), p.get('region_code', ''),
                    p.get('contact_person', ''), p.get('contact_phone', ''),
                    p.get('email', ''), p.get('address', ''),
                    p.get('main_business', ''), p.get('user_id'),
                    now, now,
                ),
            )
            self.db.commit()
        except Exception as e:
            return self._reply(msg, 'error', {'error': str(e)})

        row = self.db.execute(
            "SELECT id FROM enterprises WHERE org_code=?", (org_code,)
        ).fetchone()
        self._persist_message(msg)
        return self._reply(msg, 'enterprise_registered', {'enterprise_id': row['id'], 'status': 'pending'})

    def _handle_update(self, msg: Message) -> Message:
        p = msg.payload
        eid = p.get('enterprise_id')
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE enterprises SET
               name=COALESCE(?,name),
               nature=COALESCE(?,nature),
               industry=COALESCE(?,industry),
               contact_person=COALESCE(?,contact_person),
               contact_phone=COALESCE(?,contact_phone),
               email=COALESCE(?,email),
               address=COALESCE(?,address),
               main_business=COALESCE(?,main_business),
               updated_at=?
               WHERE id=?""",
            (
                p.get('name'), p.get('nature'), p.get('industry'),
                p.get('contact_person'), p.get('contact_phone'),
                p.get('email'), p.get('address'), p.get('main_business'),
                now, eid,
            ),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'enterprise_updated', {'enterprise_id': eid})

    def _handle_submit_data(self, msg: Message) -> Message:
        p = msg.payload
        eid = p.get('enterprise_id')
        period_id = p.get('period_id')
        baseline = p.get('baseline_employees', 0)
        survey = p.get('survey_employees', 0)

        # If survey < baseline, decrease fields are required
        if survey < baseline:
            if not p.get('decrease_type') or not p.get('main_reason'):
                return self._reply(msg, 'error', {
                    'error': '调查期人数少于建档期时，减少类型和主要原因必填'
                })

        now = datetime.now().isoformat()
        # Check for existing draft
        existing = self.db.execute(
            "SELECT id FROM submissions WHERE enterprise_id=? AND period_id=? AND status='draft'",
            (eid, period_id),
        ).fetchone()

        if existing:
            self.db.execute(
                """UPDATE submissions SET
                   baseline_employees=?, survey_employees=?,
                   decrease_type=?, main_reason=?, main_reason_desc=?,
                   secondary_reason=?, secondary_reason_desc=?,
                   third_reason=?, third_reason_desc=?,
                   status='submitted', submitted_by=?, submitted_at=?, updated_at=?
                   WHERE id=?""",
                (
                    baseline, survey,
                    p.get('decrease_type'), p.get('main_reason'), p.get('main_reason_desc'),
                    p.get('secondary_reason'), p.get('secondary_reason_desc'),
                    p.get('third_reason'), p.get('third_reason_desc'),
                    p.get('user_id'), now, now, existing['id'],
                ),
            )
            sub_id = existing['id']
        else:
            cur = self.db.execute(
                """INSERT INTO submissions
                   (enterprise_id, period_id, baseline_employees, survey_employees,
                    decrease_type, main_reason, main_reason_desc,
                    secondary_reason, secondary_reason_desc,
                    third_reason, third_reason_desc,
                    status, submitted_by, submitted_at, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,'submitted',?,?,?,?)""",
                (
                    eid, period_id, baseline, survey,
                    p.get('decrease_type'), p.get('main_reason'), p.get('main_reason_desc'),
                    p.get('secondary_reason'), p.get('secondary_reason_desc'),
                    p.get('third_reason'), p.get('third_reason_desc'),
                    p.get('user_id'), now, now, now,
                ),
            )
            sub_id = cur.lastrowid

        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'data_submitted', {'submission_id': sub_id, 'status': 'submitted'})

    def _handle_query_submissions(self, msg: Message) -> Message:
        p = msg.payload
        eid = p.get('enterprise_id')
        rows = self.db.execute(
            """SELECT s.id, sp.period_key, s.baseline_employees, s.survey_employees,
                      s.status, s.submitted_at
               FROM submissions s
               JOIN survey_periods sp ON sp.id = s.period_id
               WHERE s.enterprise_id=?
               ORDER BY s.created_at DESC""",
            (eid,),
        ).fetchall()
        data = [dict(r) for r in rows]
        return self._reply(msg, 'submission_list', {'submissions': data})

    def _handle_change_password(self, msg: Message) -> Message:
        p = msg.payload
        uid = p.get('user_id')
        new_pwd = p.get('new_password', '')
        if len(new_pwd) < 6:
            return self._reply(msg, 'error', {'error': '密码不能少于6位'})
        self.db.execute("UPDATE users SET password=? WHERE id=?", (new_pwd, uid))
        self.db.commit()
        return self._reply(msg, 'password_changed', {'user_id': uid})

    def _handle_browse_notifications(self, msg: Message) -> Message:
        rows = self.db.execute(
            "SELECT id, title, published_at FROM notifications WHERE is_deleted=0 ORDER BY published_at DESC"
        ).fetchall()
        return self._reply(msg, 'notification_list', {'notifications': [dict(r) for r in rows]})
