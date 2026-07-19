"""
LLM 预测节点 — 多 Agent 并行辩论模式

3 个专职 Agent 并行分析各自领域数据:
  - 技术面 Agent: 只看 K 线/指标
  - 基本面 Agent: 只看估值/财务
  - 舆情 Agent:   只看新闻/股吧/情感

Moderator 阅读三方观点后综合裁决，输出最终预测。

借鉴 BettaFish ForumEngine 的多 Agent 辩论模式。
"""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from loguru import logger

# 记忆系统：大师历史准确率追踪
from memory.masters.master_track import MasterTrackDB


@dataclass
class AgentView:
    """单个 Agent 的观点"""
    role: str = ""            # tech / fundamental / sentiment
    outlook: str = "中性"     # 看多/看空/中性
    confidence: str = "低"    # 高/中/低
    score: float = 0          # -10 ~ 10
    key_points: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class PredictionResult:
    """LLM 预测输出（含多 Agent 观点 + 主持人裁决）"""
    analysis_text: str = ""
    outlook: str = "中性"
    reason: str = ""
    risk_factors: List[str] = field(default_factory=list)
    positive_factors: List[str] = field(default_factory=list)

    # 各 Agent 独立观点
    tech_view: Optional[Dict] = None
    fund_view: Optional[Dict] = None
    sent_view: Optional[Dict] = None

    # 主持人多周期预测 + 操作建议
    short_term: Optional[Dict] = None
    mid_term: Optional[Dict] = None
    long_term: Optional[Dict] = None
    suggested_action: Optional[Dict] = None  # {action, reason, stop_loss, take_profit}

    # 兼容旧字段
    price_target_current: Optional[float] = None
    price_target_low: Optional[float] = None
    price_target_high: Optional[float] = None
    confidence: str = "低"

    raw_llm_output: str = ""

    # 大师决策扩展字段
    cio_decision: Optional[Dict] = None       # CIODecision.to_dict()
    employee_reports: List[Dict] = field(default_factory=list)  # 员工报告列表


class PredictionNode:
    """多 Agent 并行预测节点"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None):
        from config import settings
        self.api_key = api_key or os.environ.get('LLM_API_KEY') or getattr(settings, 'LLM_API_KEY', None) or ''
        self.base_url = base_url or os.environ.get('LLM_BASE_URL') or getattr(settings, 'LLM_BASE_URL', None) or 'https://api.openai.com/v1'
        self.model = model or os.environ.get('LLM_MODEL_NAME') or getattr(settings, 'LLM_MODEL_NAME', None) or 'gpt-4o-mini'

    @staticmethod
    def _finite_number(value) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _forecast_direction(cls, forecast: Any) -> int:
        """Return a directional vote without treating missing forecasts as bearish."""
        if not isinstance(forecast, dict):
            return 0
        direction = str(forecast.get('direction') or '').strip()
        if direction == '上涨':
            return 1
        if direction == '下跌':
            return -1
        change_pct = cls._finite_number(forecast.get('change_pct'))
        if change_pct is None:
            return 0
        return 1 if change_pct > 0 else -1 if change_pct < 0 else 0

    @classmethod
    def _quote_is_actionable(cls, state: dict, quote: dict, price: Optional[float]) -> tuple[bool, str]:
        """A trading suggestion needs a real, usable reference price."""
        if price is None or price <= 0:
            return False, "未取得有效现价"

        source = str(quote.get('source') or '').strip().lower() if isinstance(quote, dict) else ''
        if source in {'mock', '模拟数据'}:
            return False, "行情为模拟数据"

        provenance = state.get('data_provenance') if isinstance(state, dict) else None
        if not isinstance(provenance, dict):
            return True, ''
        sections = provenance.get('sections') or {}
        quote_meta = sections.get('quote') if isinstance(sections, dict) else {}
        status = str((quote_meta or {}).get('status') or '').strip().lower()
        if status in {'mock', 'unavailable'}:
            return False, "行情数据不可用" if status == 'unavailable' else "行情为模拟数据"
        return True, ''

    @classmethod
    def _sanitize_position_size(
        cls, state: dict, action: str, raw_position_size: Any
    ) -> tuple[Optional[float], str]:
        """Validate a CIO target weight against the supplied account constraints."""
        if action not in {'买入', '加仓', '持有'}:
            return None, ''

        target = cls._finite_number(raw_position_size)
        if target is None:
            return None, ''
        if target < 0:
            return None, '目标仓位无效，已隐藏'

        cap = 100.0
        quote = state.get('quote', {}) if isinstance(state, dict) else {}
        price = cls._finite_number(quote.get('price') if isinstance(quote, dict) else None) or 0.0
        shares = cls._finite_number(state.get('shares') if isinstance(state, dict) else None) or 0.0
        total_assets = cls._finite_number(state.get('total_assets') if isinstance(state, dict) else None) or 0.0
        available_cash = cls._finite_number(state.get('available_cash') if isinstance(state, dict) else None)
        if total_assets > 0 and available_cash is not None and available_cash >= 0:
            max_affordable = (shares * price + available_cash) / total_assets * 100
            cap = min(cap, max(0.0, max_affordable))

        sanitized = min(target, cap)
        if target > cap:
            return round(sanitized, 1), f'目标仓位超过资金约束，已限制为 {sanitized:.1f}%'
        return round(sanitized, 1), ''

    @classmethod
    def _apply_decision_guardrails(cls, state: dict, payload: dict) -> dict:
        """Keep model output as research, but reject contradictory trade instructions."""
        result = dict(payload or {})
        notes: list[str] = []

        outlook = str(result.get('outlook') or '中性').strip()
        if outlook not in {'看多', '看空', '中性'}:
            outlook = '中性'
            notes.append('综合方向无效，已按中性处理')

        short_term = result.get('short_term') if isinstance(result.get('short_term'), dict) else None
        mid_term = result.get('mid_term') if isinstance(result.get('mid_term'), dict) else None
        long_term = result.get('long_term') if isinstance(result.get('long_term'), dict) else None
        direction_sum = sum(cls._forecast_direction(item) for item in (short_term, mid_term, long_term))
        if outlook == '看多' and direction_sum < 0:
            outlook = '中性'
            notes.append('多周期预测整体偏弱，综合方向已降为中性')
        elif outlook == '看空' and direction_sum > 0:
            outlook = '中性'
            notes.append('多周期预测整体偏强，综合方向已降为中性')
        result['outlook'] = outlook

        quote = state.get('quote', {}) if isinstance(state, dict) else {}
        price = cls._finite_number(quote.get('price') if isinstance(quote, dict) else None)
        quote_is_actionable, quote_reason = cls._quote_is_actionable(state, quote, price)
        shares = cls._finite_number(state.get('shares') if isinstance(state, dict) else None) or 0
        has_position = shares > 0

        raw_action = result.get('suggested_action')
        raw_action = raw_action if isinstance(raw_action, dict) else {}
        action = str(raw_action.get('action') or '观望').strip()
        action = {'减仓': '减持'}.get(action, action)
        if action not in {'买入', '加仓', '持有', '减持', '卖出', '观望'}:
            action = '观望'
            notes.append('操作建议无效，已改为观望')

        short_direction = cls._forecast_direction(short_term)
        mid_direction = cls._forecast_direction(mid_term)
        if action != '观望' and not quote_is_actionable:
            action = '观望'
            notes.append(f'{quote_reason}，不能给出操作建议，已改为观望')
        elif action in {'买入', '加仓'} and (
            outlook != '看多' or short_direction < 0 or mid_direction < 0
        ):
            action = '观望'
            notes.append('买入/加仓与综合或短中期预测不一致，已改为观望')
        elif action == '加仓' and not has_position:
            action = '观望'
            notes.append('未提供持仓数量，不能给出加仓建议，已改为观望')
        elif action in {'持有', '减持', '卖出'} and not has_position:
            action = '观望'
            notes.append('未提供持仓数量，不能给出持有或卖出建议，已改为观望')

        stop_loss = cls._finite_number(raw_action.get('stop_loss'))
        take_profit = cls._finite_number(raw_action.get('take_profit'))
        if action not in {'买入', '加仓', '持有'} or price is None or price <= 0:
            stop_loss = None
            take_profit = None
        else:
            if stop_loss is not None and not (0 < stop_loss < price):
                stop_loss = None
                notes.append('止损价不在现价下方，已隐藏')
            if take_profit is not None and take_profit <= price:
                take_profit = None
                notes.append('止盈价不在现价上方，已隐藏')

        action_reason = str(raw_action.get('reason') or '').strip()
        if notes:
            action_reason = f"{action_reason}；系统校验：{'；'.join(notes)}".strip('；')
            result['confidence'] = '低'
        result['suggested_action'] = {
            'action': action,
            'reason': action_reason or '仅供研究参考，请结合仓位、风险承受能力和成交条件手工判断。',
            'stop_loss': stop_loss,
            'take_profit': take_profit,
        }

        low = cls._finite_number(result.get('price_target_low'))
        high = cls._finite_number(result.get('price_target_high'))
        if price is None or price <= 0:
            low = high = None
        else:
            low = low if low is not None and 0 < low < price else None
            high = high if high is not None and high > price else None
        result['price_target_low'] = low
        result['price_target_high'] = high
        return result

    # ── 主入口 ──

    def predict(self, state: dict) -> PredictionResult:
        if self.api_key:
            return self._multi_agent_predict(state)
        else:
            return self._rule_predict(state)

    # ── 多 Agent 并行 ──

    def _multi_agent_predict(self, state: dict) -> PredictionResult:
        """3 Agent 并行分析 → Moderator 综合裁决"""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # 并行调用 3 个 Agent
        agents = {
            'tech':         (self._tech_prompt, self._build_tech_data(state)),
            'fundamental':  (self._fund_prompt, self._build_fund_data(state)),
            'sentiment':    (self._sent_prompt, self._build_sent_data(state)),
        }

        views: Dict[str, AgentView] = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._call_agent, client, role, prompt, data): role
                for role, (prompt, data) in agents.items()
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    views[role] = future.result(timeout=30)
                except Exception as e:
                    logger.warning(f"Agent [{role}] 失败: {e}")
                    views[role] = AgentView(role=role, outlook="中性", confidence="低",
                                            key_points=[f"Agent 调用失败: {str(e)[:50]}"])

        # 主持人综合裁决
        debate_text = self._format_debate(state, views)
        final = self._apply_decision_guardrails(
            state, self._call_moderator(client, state, views, debate_text)
        )

        # 组装结果
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0

        return PredictionResult(
            analysis_text=final.get('analysis_text', ''),
            outlook=final.get('outlook', views.get('tech', AgentView()).outlook),
            reason=final.get('reason', ''),
            risk_factors=final.get('risk_factors', []),
            positive_factors=final.get('positive_factors', []),
            tech_view=self._view_to_dict(views.get('tech')),
            fund_view=self._view_to_dict(views.get('fundamental')),
            sent_view=self._view_to_dict(views.get('sentiment')),
            short_term=final.get('short_term'),
            mid_term=final.get('mid_term'),
            long_term=final.get('long_term'),
            suggested_action=final.get('suggested_action'),
            price_target_current=price,
            price_target_low=final.get('price_target_low'),
            price_target_high=final.get('price_target_high'),
            confidence=final.get('confidence', '低'),
            raw_llm_output=final.get('raw', ''),
        )

    # ── 大师决策模式 (8 员工 + CIO) ──

    def predict_with_master(self, state: dict, master_key: str) -> PredictionResult:
        """
        大师决策模式: 6~8 名员工并行分析 → 大师 CIO 最终决策。

        Args:
            state: AnalysisState.to_dict() 输出
            master_key: 大师标识 (buffett/graham/fisher/lynch/templeton/soros/dalio)

        Returns:
            PredictionResult (含员工报告 + CIO 决策)
        """
        from analysis.agents.valuation_agent import ValuationAgent
        from analysis.agents.fundamental_agent import FundamentalAgent
        from analysis.agents.technical_agent import TechnicalAgent
        from analysis.agents.sentiment_agent import SentimentAgent
        from analysis.agents.risk_manager import RiskManager
        from analysis.agents.overseer import Overseer
        from analysis.agents.macro_agent import MacroAgent
        from analysis.agents.policy_agent import PolicyAgent
        from analysis.agents.cio import CIOAgent

        # 初始化 8 名员工
        employees = [
            MacroAgent(api_key=self.api_key, base_url=self.base_url, model=self.model),
            PolicyAgent(api_key=self.api_key, base_url=self.base_url, model=self.model),
            ValuationAgent(api_key=self.api_key, base_url=self.base_url, model=self.model),
            FundamentalAgent(api_key=self.api_key, base_url=self.base_url, model=self.model),
            TechnicalAgent(api_key=self.api_key, base_url=self.base_url, model=self.model),
            SentimentAgent(api_key=self.api_key, base_url=self.base_url, model=self.model),
            RiskManager(api_key=self.api_key, base_url=self.base_url, model=self.model),
        ]
        overseer = Overseer(api_key=self.api_key, base_url=self.base_url, model=self.model)

        # 并行执行前 7 名员工 (overseer 依赖其他员工输出)
        def _run_employee(emp, st):
            try:
                return emp.analyze(st)
            except Exception as e:
                logger.warning(f"员工 [{emp.role}] 失败: {e}")
                from analysis.agents.base import EmployeeReport
                return EmployeeReport(employee_id=emp.employee_id, role=emp.role,
                                      department=emp.department,
                                      outlook="中性", confidence="低",
                                      key_points=[f"报告生成失败: {str(e)[:80]}"],
                                      error=str(e)[:200])

        reports = []
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {executor.submit(_run_employee, emp, state): emp for emp in employees}
            for future in as_completed(futures):
                emp = futures[future]
                try:
                    r = future.result(timeout=45)
                    reports.append(r)
                    logger.info(f"员工 [{emp.role}] 完成: {r.outlook} (score={r.score})")
                except Exception as e:
                    logger.warning(f"员工 [{emp.role}] 超时/异常: {e}")
                    from analysis.agents.base import EmployeeReport
                    reports.append(EmployeeReport(
                        employee_id=emp.employee_id, role=emp.role, department=emp.department,
                        outlook="中性", confidence="低",
                        key_points=[f"报告生成超时: {str(e)[:80]}"],
                        error=str(e)[:200]))

        # 独立监察员读取其他员工报告
        overseer_report = _run_employee(overseer, state)
        # Overseer needs other reports for context - re-run with reports
        try:
            overseer_report = overseer.analyze(state, other_reports=reports)
        except Exception as e:
            logger.warning(f"监察员失败: {e}")
        reports.append(overseer_report)
        logger.info(f"员工 [监察员] 完成")

        # CIO 最终决策（带历史准确率追踪）
        cio = CIOAgent(api_key=self.api_key, base_url=self.base_url, model=self.model)

        # 查询大师历史准确率
        track_context = MasterTrackDB().get_prompt_context(
            master_key=master_key,
            symbol=state.get('symbol', ''),
            limit=5,
        )

        cio_decision = cio.decide(master_key, reports, state, track_context=track_context)
        logger.info(f"CIO [{cio_decision.master_name}] 决策: {cio_decision.decision_summary[:60]}")

        # 组装 PredictionResult
        return self._build_master_result(state, cio_decision, reports)

    def _build_master_result(self, state: dict, cio_decision,
                               reports: list) -> PredictionResult:
        """将 CIO 决策 + 员工报告合并为 PredictionResult"""
        from analysis.agents.base import CIODecision
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0

        order = cio_decision.order or {}
        master_direction = cio_decision.short_term.get('direction', '中性') if cio_decision.short_term else '中性'
        master_outlook = {'上涨': '看多', '下跌': '看空', '震荡': '中性'}.get(master_direction, master_direction)
        guarded = self._apply_decision_guardrails(state, {
            'outlook': master_outlook,
            'confidence': cio_decision.decision_quality.get('confidence', '低') if cio_decision.decision_quality else '低',
            'short_term': cio_decision.short_term,
            'mid_term': cio_decision.mid_term,
            'long_term': cio_decision.long_term,
            'suggested_action': {
                'action': order.get('action', '观望'),
                'reason': order.get('entry_conditions', ''),
                'stop_loss': order.get('stop_loss', {}).get('level') if isinstance(order.get('stop_loss'), dict) else None,
                'take_profit': order.get('take_profit', {}).get('level_1') if isinstance(order.get('take_profit'), dict) else None,
            },
            'price_target_low': cio_decision.bear_case.get('target') if cio_decision.bear_case else None,
            'price_target_high': cio_decision.bull_case.get('target') if cio_decision.bull_case else None,
        })
        # The frontend also renders the persisted CIO object directly. Keep it
        # aligned with the guarded summary so two parts of one report cannot
        # show conflicting trade instructions.
        guarded_order = dict(order)
        position_size_pct, position_note = self._sanitize_position_size(
            state, guarded['suggested_action']['action'], order.get('position_size_pct')
        )
        guarded_order.update({
            'action': guarded['suggested_action']['action'],
            'position_size_pct': position_size_pct,
            'entry_conditions': guarded['suggested_action']['reason'],
            'stop_loss': {
                **(order.get('stop_loss') if isinstance(order.get('stop_loss'), dict) else {}),
                'level': guarded['suggested_action']['stop_loss'] or 0,
            },
            'take_profit': {
                **(order.get('take_profit') if isinstance(order.get('take_profit'), dict) else {}),
                'level_1': guarded['suggested_action']['take_profit'] or 0,
            },
        })
        if position_note:
            existing_note = str(guarded_order.get('position_note') or '').strip()
            guarded_order['position_note'] = f'{existing_note}；系统校验：{position_note}'.strip('；')
            guarded['confidence'] = '低'
        cio_decision.order = guarded_order

        result = PredictionResult(
            analysis_text=cio_decision.decision_summary,
            outlook=guarded['outlook'],
            reason=cio_decision.decision_summary,
            risk_factors=[r for report in reports if not report.error for r in (report.risks or [])[:2]],
            positive_factors=[r for report in reports if not report.error for r in (report.key_points or [])[:2]],
            # 多 Agent 观点 (前3个映射到 legacy 字段)
            tech_view=self._emp_to_view_dict(reports, 'technical'),
            fund_view=self._emp_to_view_dict(reports, 'fundamental'),
            sent_view=self._emp_to_view_dict(reports, 'sentiment'),
            # 多周期预测
            short_term=cio_decision.short_term,
            mid_term=cio_decision.mid_term,
            long_term=cio_decision.long_term,
            # 操作建议
            suggested_action=guarded['suggested_action'],
            price_target_current=price,
            price_target_low=guarded['price_target_low'],
            price_target_high=guarded['price_target_high'],
            confidence=guarded.get('confidence', '低'),
            raw_llm_output=cio_decision.raw_llm_output,
        )

        # 附加 CIO 决策和员工报告到 result (通过非标准字段)
        result.cio_decision = cio_decision.to_dict() if isinstance(cio_decision, CIODecision) else {}
        result.employee_reports = [
            {
                'employee_id': r.employee_id, 'role': r.role, 'department': r.department,
                'outlook': r.outlook, 'confidence': r.confidence, 'score': r.score,
                'key_points': r.key_points, 'risks': r.risks, 'error': r.error,
            }
            for r in reports
        ]

        return result

    @staticmethod
    def _emp_to_view_dict(reports: list, emp_id: str) -> Optional[dict]:
        """从员工报告列表中查找指定 ID 的报告，转为 legacy view dict 格式"""
        for r in reports:
            if r.employee_id == emp_id and not r.error:
                return {
                    'role': r.role, 'outlook': r.outlook, 'confidence': r.confidence,
                    'score': r.score, 'key_points': r.key_points, 'risks': r.risks,
                }
        return None

    # ── Agent 调用 ──

    def _call_agent(self, client, role: str, prompt: str, data: str) -> AgentView:
        full_prompt = f"{prompt}\n\n## 分析数据\n{data}"
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"你是A股{role}分析专家。请仅基于提供的数据给出独立判断。输出严格JSON。"},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        d = self._parse_json(raw)
        return AgentView(
            role=role,
            outlook=d.get('outlook', '中性'),
            confidence=d.get('confidence', '低'),
            score=float(d.get('score', 0)),
            key_points=d.get('key_points', []),
            risks=d.get('risks', []),
            raw_output=raw,
        )

    def _call_moderator(self, client, state: dict, views: Dict[str, AgentView], debate_text: str) -> dict:
        """主持人阅读三方辩论后给出最终判断"""
        prompt = self._moderator_prompt(state, debate_text)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是A股投资委员会主席。三位分析师（技术面、基本面、舆情）已给出独立判断。请你审阅三方观点，辩论、裁决，输出最终预测JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            result = self._parse_json(raw)
            result['raw'] = raw
            return result
        except Exception as e:
            logger.warning(f"Moderator 调用失败: {e}")
            # 降级：取多数 Agent 的观点
            outlooks = [v.outlook for v in views.values()]
            majority = max(set(outlooks), key=outlooks.count)
            return {
                'outlook': majority, 'confidence': '低',
                'analysis_text': f"主持人调用失败({e})，取多数Agent观点: {majority}",
                'reason': f"Agent投票: " + ", ".join(f"{r}={o}" for r, o in zip(views.keys(), outlooks)),
            }

    # ── 提示词 ──

    @property
    def _tech_prompt(self) -> str:
        return """你是技术分析专家。仅根据K线指标给出独立判断，不要考虑基本面或消息面。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": -5,
  "key_points": ["MACD死叉且柱状线扩大，短期动能偏空", "RSI=19进入超卖区，技术性反弹概率增加"],
  "risks": ["均线空头排列，趋势尚未扭转", "若跌破布林下轨可能加速下跌"]
}

评分规则:
- 每项看多信号+1分，强看多+2分；看空-1分，强看空-2分
- 最终score为各项加总，范围约-10~10
- outlook: score>2→看多, score<-2→看空, 否则中性
- 仅输出JSON，不要额外文字"""

    @property
    def _fund_prompt(self) -> str:
        return """你是基本面/估值分析专家。仅根据财务数据和PE分位给出独立判断，不要考虑技术面或消息面。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 3,
  "key_points": ["PE处于历史1%分位，远低于均值，估值极低", "ROE=10.6%盈利能力稳健"],
  "risks": ["行业景气度下行可能压制估值修复", "营收增速放缓需关注"]
}

评分规则:
- PE分位<10%: +3分; <30%: +1分; >70%: -1分; >90%: -3分
- ROE>15%: +2分; ROE>10%: +1分; ROE<5%: -1分
- EPS同比正增长: +1分; 负增长: -1分
- outlook: score>2→看多, score<-2→看空, 否则中性
- 仅输出JSON，不要额外文字"""

    @property
    def _sent_prompt(self) -> str:
        return """你是舆情分析专家。仅根据新闻和股吧数据给出独立判断，不要考虑技术面或基本面。

输出JSON:
{
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "score": 2,
  "key_points": ["茅台30亿回购完成，注销股份提振信心", "白酒板块集体下挫，市场情绪偏谨慎"],
  "risks": ["主力资金净流出", "股吧看空帖子多于看多"]
}

评分规则:
- 情感avg>0.3: +2分; >0.1: +1分; <-0.1: -1分; <-0.3: -2分
- 正面占比>50%: +1分; 负面占比>50%: -1分
- outlook: score>2→看多, score<-2→看空, 否则中性
- 仅输出JSON，不要额外文字"""

    def _moderator_prompt(self, state: dict, debate_text: str) -> str:
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        cost = state.get('cost_price', 0) or 0
        val = state.get('valuation_level', '正常')
        reference_price = state.get('suggested_buy_price')
        reference_text = f"{reference_price:.2f}" if isinstance(reference_price, (int, float)) and reference_price > 0 else "暂不适用"
        valuation_note = state.get('valuation_note') or ""

        cost_lines = ""
        if cost > 0:
            pnl_pct = (price - cost) / cost * 100
            cost_lines = f"""
## 持仓信息
- 成本价: {cost}  现价: {price}
- 浮动盈亏: {pnl_pct:+.1f}%
- 估值等级: {val}  估值参考价: {reference_text}
{f'- 估值说明: {valuation_note}' if valuation_note else ''}
"""
        else:
            cost_lines = f"""
## 参考价位
- 现价: {price}  估值等级: {val}
- 估值参考价: {reference_text}
{f'- 估值说明: {valuation_note}' if valuation_note else ''}
"""

        return f"""## 股票信息
{state.get('stock_name', '')}({state.get('symbol', '')})  现价: {price}
{cost_lines}
## 三位分析师独立观点

{debate_text}

## 你的任务

作为投资委员会主席，请审阅上述三方观点后:

1. **指出共识** — 三位分析师在哪些判断上一致？
2. **指出分歧** — 哪些判断相互矛盾？你更认同一方的理由是什么？
3. **综合裁决** — 给出最终的多空判断和置信度
4. **多周期预测** — 综合技术面(短期)、估值(中长期)、舆情(情绪面)给出短/中/长期涨跌预测
5. **操作建议** — 综合{'持仓盈亏、' if cost else ''}估值分位、技术信号，给出具体操作建议

输出JSON:
{{
  "analysis_text": "综合三位分析师观点的完整分析(200字内)",
  "outlook": "看多/看空/中性",
  "confidence": "高/中/低",
  "reason": "核心裁决逻辑(80字内)",
  "short_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 3.5,
    "confidence": "高/中/低",
    "reason": "1~2周预测依据(40字内)"
  }},
  "mid_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 8.0,
    "confidence": "高/中/低",
    "reason": "1~3月预测依据(40字内)"
  }},
  "long_term": {{
    "direction": "上涨/下跌/震荡",
    "change_pct": 15.0,
    "confidence": "高/中/低",
    "reason": "6~12月预测依据(40字内)"
  }},
   "suggested_action": {{
     "action": "买入/加仓/持有/减持/卖出/观望",
     "reason": "操作理由(60字内)",
     "stop_loss": null,
     "take_profit": null
   }},
   "price_target_low": null,
   "price_target_high": null,
  "risk_factors": ["风险1", "风险2"],
  "positive_factors": ["积极因素1", "积极因素2"]
}}

注意:
- short_term.change_pct: 预计1-2周内的涨跌幅度，正数上涨负数下跌
- mid_term.change_pct: 预计1-3月内的涨跌幅度，侧重估值回归
- long_term.change_pct: 预计6-12月内的涨跌幅度，侧重基本面和行业趋势
- 没有明确的技术支撑/压力位、风险依据或情景测算时，止损、止盈和目标价必须为 null；不得使用固定百分比套算。
- 买入或加仓只适用于综合判断看多且短期、中期不预测下跌的情况；其余情况使用观望。"""

    # ── 数据构造 (每个 Agent 只看自己的领域) ──

    def _build_tech_data(self, state: dict) -> str:
        ti = state.get('technical_indicators', {}) or {}
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0
        lines = [
            f"现价: {price}  涨跌: {q.get('change_pct','?')}%",
            f"RSI(14): {ti.get('rsi_14','?')}  MACD柱: {ti.get('macd_hist','?')}",
            f"KDJ: {ti.get('kdj_k','?')}/{ti.get('kdj_d','?')}/{ti.get('kdj_j','?')}",
            f"MA5:{ti.get('ma5','?')} MA10:{ti.get('ma10','?')} MA20:{ti.get('ma20','?')} MA60:{ti.get('ma60','?')}",
            f"布林: {ti.get('boll_lower','?')} ~ {ti.get('boll_middle','?')} ~ {ti.get('boll_upper','?')}",
            f"量比: {ti.get('volume_ratio','?')}  振幅: {ti.get('amplitude','?')}%",
        ]
        return "\n".join(lines)

    def _build_fund_data(self, state: dict) -> str:
        fs = state.get('financial_summary', {}) or {}
        q = state.get('quote', {}) or {}
        pe_pct = state.get('valuation_percentile')
        pe_pct_str = f"{pe_pct:.1f}%" if pe_pct is not None else "N/A"
        valuation_note = state.get('valuation_note') or ""
        lines = [
            f"PE: {q.get('pe','?')}  PB: {q.get('pb','?')}  市值: {q.get('market_cap','?')}亿",
            f"估值等级: {state.get('valuation_level','?')} (PE分位: {pe_pct_str})",
            f"EPS: {fs.get('eps','?')}  ROE: {fs.get('roe','?')}%  "
            f"经营现金流: {fs.get('operating_cash_flow','N/A')}亿  "
            f"自由现金流: {fs.get('free_cash_flow','N/A')}亿",
            f"营收: {fs.get('revenue','?')}亿  净利: {fs.get('net_profit','?')}亿",
            f"股息率: {fs.get('dividend_yield') or q.get('dividend_yield','N/A')}%",
            f"毛利率: {fs.get('gross_margin','?')}%  负债率: {fs.get('debt_ratio','?')}%",
        ]
        if valuation_note:
            lines.append(f"估值说明: {valuation_note}")
        return "\n".join(lines)

    def _build_sent_data(self, state: dict) -> str:
        sn = state.get('sentiment_news', {}) or {}
        sg = state.get('sentiment_guba', {}) or {}
        bull_news = state.get('important_bullish_news', []) or []
        bear_news = state.get('important_bearish_news', []) or []
        bull_guba = state.get('important_bullish_guba', []) or []
        bear_guba = state.get('important_bearish_guba', []) or []

        lines = [
            f"新闻: {sn.get('total_count',0)}条  正面{sn.get('positive_count',0)}  负面{sn.get('negative_count',0)}  avg={sn.get('avg_score',0):.2f}",
            f"股吧: {sg.get('total_count',0)}条  正面{sg.get('positive_count',0)}  负面{sg.get('negative_count',0)}  avg={sg.get('avg_score',0):.2f}",
            "",
            "利好:",
        ]
        for n in bull_news[:3]:
            lines.append(f"  + {n['title']}")
        for g in bull_guba[:2]:
            lines.append(f"  + [股吧] {g['title']}")
        lines.append("")
        lines.append("利空:")
        for n in bear_news[:3]:
            lines.append(f"  - {n['title']}")
        for g in bear_guba[:2]:
            lines.append(f"  - [股吧] {g['title']}")
        return "\n".join(lines)

    def _format_debate(self, state: dict, views: Dict[str, AgentView]) -> str:
        """格式化三方观点供主持人阅读"""
        names = {'tech': '技术面分析师', 'fundamental': '基本面分析师', 'sentiment': '舆情分析师'}
        parts = []
        for role, v in views.items():
            name = names.get(role, role)
            parts.append(
                f"### {name}\n"
                f"判断: {v.outlook}  置信度: {v.confidence}  评分: {v.score}\n"
                f"看多理由:\n" + "\n".join(f"  - {p}" for p in v.key_points) + "\n"
                f"风险点:\n" + "\n".join(f"  - {r}" for r in v.risks)
            )
        return "\n\n".join(parts)

    # ── 规则降级 ──

    def _rule_predict(self, state: dict) -> PredictionResult:
        signals = state.get('signals', {}) or {}
        score = signals.get('score', 0)
        q = state.get('quote', {}) or {}
        price = q.get('price', 0) if isinstance(q, dict) else 0

        if score > 2:
            outlook = "看多"
        elif score < -2:
            outlook = "看空"
        else:
            outlook = "中性"

        conf = "高" if abs(score) > 4 else "中" if abs(score) > 2 else "低"

        return PredictionResult(
            outlook=outlook,
            confidence=conf,
            analysis_text=f"## {outlook}信号 (规则模式)\n\n综合评分 {score:.1f}\n\n规则模式不生成目标价。",
            reason=f"规则引擎: 综合评分{score:.1f}；未生成目标价",
            price_target_current=price,
            # The rule fallback has no defensible price-target model. Never fabricate
            # a price range by applying fixed percentages to the current quote.
            price_target_low=None,
            price_target_high=None,
            risk_factors=[],
            positive_factors=[],
        )

    # ── 工具 ──

    @staticmethod
    def _view_to_dict(v: Optional[AgentView]) -> Optional[dict]:
        if v is None:
            return None
        return {
            'role': v.role, 'outlook': v.outlook, 'confidence': v.confidence,
            'score': v.score, 'key_points': v.key_points, 'risks': v.risks,
        }

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            raw = raw.strip()
            if raw.startswith('```json'):
                raw = raw.split('```json')[1].split('```')[0]
            elif raw.startswith('```'):
                raw = raw.split('```')[1].split('```')[0]
            return json.loads(raw)
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"JSON 解析失败: {raw[:100]}")
            return {}
