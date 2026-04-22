"""
ProvinceAgent – represents the provincial-level bureau (省局).

Responsibilities:
  - Approve / return city-approved submissions
  - Aggregate data and generate reports
  - Manage enterprise filing (approve/reject)
  - Multi-dimensional analysis
  - Forward aggregated data to ministry (simulated)
  - Manage survey periods and notifications
"""
from __future__ import annotations
from datetime import datetime
from .base_agent import BaseAgent, Message


class ProvinceAgent(BaseAgent):
    """Agent acting on behalf of the provincial labour bureau."""

    def _register_handlers(self):
        self.register('approve_enterprise', self._handle_approve_enterprise)
        self.register('reject_enterprise', self._handle_reject_enterprise)
        self.register('list_filed_enterprises', self._handle_list_filed)
        self.register('approve_submission', self._handle_approve_submission)
        self.register('return_submission', self._handle_return_submission)
        self.register('aggregate_report', self._handle_aggregate)
        self.register('trend_analysis', self._handle_trend)
        self.register('comparative_analysis', self._handle_comparative)
        self.register('publish_notification', self._handle_publish_notification)
        self.register('delete_notification', self._handle_delete_notification)
        self.register('create_survey_period', self._handle_create_period)
        self.register('query_data', self._handle_query_data)
        self.register('forward_to_ministry', self._handle_forward)

    # ------------------------------------------------------------------
    # Enterprise filing
    # ------------------------------------------------------------------

    def _handle_approve_enterprise(self, msg: Message) -> Message:
        p = msg.payload
        eid = p.get('enterprise_id')
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE enterprises SET status='approved', updated_at=? WHERE id=? AND status='pending'",
            (now, eid),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'enterprise_approved', {'enterprise_id': eid})

    def _handle_reject_enterprise(self, msg: Message) -> Message:
        p = msg.payload
        eid = p.get('enterprise_id')
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE enterprises SET status='rejected', updated_at=? WHERE id=?",
            (now, eid),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'enterprise_rejected', {'enterprise_id': eid})

    def _handle_list_filed(self, msg: Message) -> Message:
        p = msg.payload
        region = p.get('region_code', '')
        period = p.get('period_key', '')
        query = "SELECT id, org_code, name, nature, industry, region_code, status FROM enterprises WHERE 1=1"
        params = []
        if region:
            query += " AND region_code LIKE ?"
            params.append(region + '%')
        if period:
            # filter enterprises that have submitted for this period
            query = (
                "SELECT DISTINCT e.id, e.org_code, e.name, e.nature, e.industry, "
                "e.region_code, e.status FROM enterprises e "
                "JOIN submissions s ON s.enterprise_id=e.id "
                "JOIN survey_periods sp ON sp.id=s.period_id "
                "WHERE sp.period_key=?"
            )
            params = [period]
            if region:
                query += " AND e.region_code LIKE ?"
                params.append(region + '%')
        rows = self.db.execute(query, params).fetchall()
        return self._reply(msg, 'filed_enterprise_list', {'enterprises': [dict(r) for r in rows]})

    # ------------------------------------------------------------------
    # Submission review
    # ------------------------------------------------------------------

    def _handle_approve_submission(self, msg: Message) -> Message:
        p = msg.payload
        sub_id = p.get('submission_id')
        user_id = p.get('user_id')
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE submissions
               SET status='province_approved', province_reviewed_by=?,
                   province_reviewed_at=?, updated_at=?
               WHERE id=? AND status='city_approved'""",
            (user_id, now, now, sub_id),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'submission_approved', {'submission_id': sub_id, 'level': 'province'})

    def _handle_return_submission(self, msg: Message) -> Message:
        p = msg.payload
        sub_id = p.get('submission_id')
        user_id = p.get('user_id')
        reason = p.get('return_reason', '')
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE submissions
               SET status='returned', province_reviewed_by=?,
                   province_reviewed_at=?, return_reason=?, updated_at=?
               WHERE id=? AND status='city_approved'""",
            (user_id, now, reason, now, sub_id),
        )
        self.db.commit()
        self._persist_message(msg)
        return self._reply(msg, 'submission_returned', {'submission_id': sub_id, 'return_reason': reason})

    # ------------------------------------------------------------------
    # Analysis and aggregation
    # ------------------------------------------------------------------

    def _handle_aggregate(self, msg: Message) -> Message:
        p = msg.payload
        period_key = p.get('period_key')
        rows = self.db.execute(
            """SELECT e.region_code,
                      COUNT(*) AS enterprise_count,
                      SUM(s.baseline_employees) AS total_baseline,
                      SUM(s.survey_employees)   AS total_survey,
                      SUM(s.baseline_employees - s.survey_employees) AS total_decrease
               FROM submissions s
               JOIN enterprises e ON e.id = s.enterprise_id
               JOIN survey_periods sp ON sp.id = s.period_id
               WHERE sp.period_key=? AND s.status='province_approved'
               GROUP BY e.region_code""",
            (period_key,),
        ).fetchall()
        report = {
            'period_key': period_key,
            'by_region': [dict(r) for r in rows],
            'summary': {
                'total_enterprises': sum(r['enterprise_count'] for r in rows),
                'total_baseline': sum(r['total_baseline'] or 0 for r in rows),
                'total_survey': sum(r['total_survey'] or 0 for r in rows),
                'total_decrease': sum(r['total_decrease'] or 0 for r in rows),
            },
        }
        return self._reply(msg, 'aggregate_report', {'report': report})

    def _handle_trend(self, msg: Message) -> Message:
        """Return trend across multiple consecutive periods."""
        p = msg.payload
        periods = p.get('periods', [])  # list of period_keys
        results = []
        for pk in periods:
            row = self.db.execute(
                """SELECT sp.period_key,
                          COUNT(*) AS enterprise_count,
                          SUM(s.survey_employees) AS total_survey,
                          SUM(s.baseline_employees) AS total_baseline
                   FROM submissions s
                   JOIN survey_periods sp ON sp.id=s.period_id
                   WHERE sp.period_key=? AND s.status='province_approved'""",
                (pk,),
            ).fetchone()
            if row:
                results.append(dict(row))
        return self._reply(msg, 'trend_data', {'trend': results})

    def _handle_comparative(self, msg: Message) -> Message:
        """Compare two survey periods."""
        p = msg.payload
        period_a = p.get('period_a')
        period_b = p.get('period_b')
        dimension = p.get('dimension', 'region')  # region / nature / industry

        col_map = {'region': 'e.region_code', 'nature': 'e.nature', 'industry': 'e.industry'}
        group_col = col_map.get(dimension, 'e.region_code')

        def _fetch(pk):
            rows = self.db.execute(
                f"""SELECT {group_col} AS dim,
                          COUNT(*) AS cnt,
                          SUM(s.baseline_employees) AS baseline,
                          SUM(s.survey_employees) AS survey
                   FROM submissions s
                   JOIN enterprises e ON e.id=s.enterprise_id
                   JOIN survey_periods sp ON sp.id=s.period_id
                   WHERE sp.period_key=? AND s.status='province_approved'
                   GROUP BY {group_col}""",
                (pk,),
            ).fetchall()
            return [dict(r) for r in rows]

        return self._reply(msg, 'comparative_data', {
            'period_a': {'key': period_a, 'data': _fetch(period_a)},
            'period_b': {'key': period_b, 'data': _fetch(period_b)},
        })

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _handle_publish_notification(self, msg: Message) -> Message:
        p = msg.payload
        title = p.get('title', '')
        content = p.get('content', '')
        user_id = p.get('user_id')
        if len(title) > 50:
            return self._reply(msg, 'error', {'error': '通知标题不超过50字'})
        if len(content) > 2000:
            return self._reply(msg, 'error', {'error': '通知内容不超过2000字'})
        now = datetime.now().isoformat()
        cur = self.db.execute(
            "INSERT INTO notifications(title, content, published_by, published_at) VALUES (?,?,?,?)",
            (title, content, user_id, now),
        )
        self.db.commit()
        return self._reply(msg, 'notification_published', {'notification_id': cur.lastrowid})

    def _handle_delete_notification(self, msg: Message) -> Message:
        p = msg.payload
        nid = p.get('notification_id')
        self.db.execute("UPDATE notifications SET is_deleted=1 WHERE id=?", (nid,))
        self.db.commit()
        return self._reply(msg, 'notification_deleted', {'notification_id': nid})

    # ------------------------------------------------------------------
    # Survey period management
    # ------------------------------------------------------------------

    def _handle_create_period(self, msg: Message) -> Message:
        p = msg.payload
        period_key = p.get('period_key')
        start_date = p.get('start_date')
        end_date = p.get('end_date')
        user_id = p.get('user_id')
        try:
            cur = self.db.execute(
                "INSERT INTO survey_periods(period_key, start_date, end_date, created_by) VALUES (?,?,?,?)",
                (period_key, start_date, end_date, user_id),
            )
            self.db.commit()
        except Exception as e:
            return self._reply(msg, 'error', {'error': str(e)})
        return self._reply(msg, 'period_created', {'period_id': cur.lastrowid, 'period_key': period_key})

    # ------------------------------------------------------------------
    # Data query / export
    # ------------------------------------------------------------------

    def _handle_query_data(self, msg: Message) -> Message:
        p = msg.payload
        filters = []
        params = []
        if p.get('period_key'):
            filters.append("sp.period_key=?")
            params.append(p['period_key'])
        if p.get('region_code'):
            filters.append("e.region_code LIKE ?")
            params.append(p['region_code'] + '%')
        if p.get('enterprise_name'):
            filters.append("e.name LIKE ?")
            params.append('%' + p['enterprise_name'] + '%')
        if p.get('status'):
            filters.append("s.status=?")
            params.append(p['status'])
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        rows = self.db.execute(
            f"""SELECT e.name, e.org_code, e.region_code, sp.period_key,
                       s.baseline_employees, s.survey_employees, s.status, s.submitted_at
                FROM submissions s
                JOIN enterprises e ON e.id=s.enterprise_id
                JOIN survey_periods sp ON sp.id=s.period_id
                {where}
                ORDER BY s.submitted_at DESC""",
            params,
        ).fetchall()
        return self._reply(msg, 'query_result', {'records': [dict(r) for r in rows]})

    def _handle_forward(self, msg: Message) -> Message:
        """Simulate forwarding province-approved data to the ministry."""
        p = msg.payload
        period_key = p.get('period_key')
        row = self.db.execute(
            """SELECT COUNT(*) AS cnt, SUM(s.survey_employees) AS total
               FROM submissions s
               JOIN survey_periods sp ON sp.id=s.period_id
               WHERE sp.period_key=? AND s.status='province_approved'""",
            (period_key,),
        ).fetchone()
        # In a real system this would call the national API
        return self._reply(msg, 'forwarded_to_ministry', {
            'period_key': period_key,
            'records_forwarded': row['cnt'],
            'total_employees': row['total'],
            'forwarded_at': datetime.now().isoformat(),
        })
