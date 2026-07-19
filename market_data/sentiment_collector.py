"""
舆情情感采集器
使用项目内财经关键词规则，对新闻和股吧内容进行情感打分。
"""
import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger


# ========== 数据结构 ==========

@dataclass
class SentimentScore:
    """单条文本的情感得分"""
    text: str
    label: str            # 非常负面 / 负面 / 中性 / 正面 / 非常正面
    confidence: float     # 置信度 0-1
    score: float          # 综合得分 -1.0 ~ 1.0
    success: bool = True


@dataclass
class SentimentSummary:
    """情感分析汇总"""
    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_score: float           # 平均得分 -1.0 ~ 1.0
    positive_ratio: float      # 正面比例 0-1
    negative_ratio: float      # 负面比例 0-1
    scores: List[SentimentScore] = field(default_factory=list)


# ========== 情感分析器 ==========

class SentimentCollector:
    """项目内规则驱动的舆情情感采集器。"""

    def __init__(self, enable_sentiment: bool = True):
        self.enable_sentiment = enable_sentiment

    def analyze_text(self, text: str) -> SentimentScore:
        """使用财经关键词规则分析单条文本。"""
        if not text or not text.strip():
            return SentimentScore(text=text, label='中性', confidence=1.0, score=0.0)
        return self._rule_based_sentiment(text)

    def analyze_batch(self, texts: List[str]) -> SentimentSummary:
        """批量分析情感"""
        scores = [self.analyze_text(t) for t in texts]
        return self._build_summary(scores)

    def analyze_news(self, news_list: List[Dict[str, str]]) -> SentimentSummary:
        """分析新闻列表情感"""
        texts = []
        for item in news_list:
            title = item.get('title', '') or item.get('Title', '')
            # 优先用内容，没有则用标题
            content = item.get('content', '') or item.get('Content', '')
            texts.append(content or title)

        return self.analyze_batch(texts)

    def analyze_guba_posts(self, posts: List[Any]) -> SentimentSummary:
        """分析股吧帖子情感"""
        texts = []
        for post in posts:
            if hasattr(post, 'title'):
                texts.append(post.title)
            elif isinstance(post, dict):
                texts.append(post.get('title', ''))
            else:
                texts.append(str(post))

        return self.analyze_batch(texts)

    # ---- 内部工具 ----

    @staticmethod
    def _rule_based_sentiment(text: str) -> SentimentScore:
        """规则降级：基于关键词的简单情感判断"""
        text_lower = text.lower()

        positive_words = ['涨', '涨停', '利好', '突破', '拉升', '买入', '推荐',
                          '增长', '盈利', '看好', '牛市', '反弹', '放量', '强势',
                          '降准', '降息', '放水', '宽松', '复苏',
                          'up', 'bullish', 'buy', 'positive', 'growth', 'rally']
        negative_words = ['跌', '跌停', '利空', '暴跌', '减持', '卖出', '风险',
                          '亏损', '暴雷', '崩盘', '熊市', '破位', '缩量', '弱势',
                          '加息', '收紧', '通缩', '违约', '退市', 'st', 'st\n*',
                          'down', 'bearish', 'sell', 'negative', 'crash', 'risk']

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            label = '正面'
            confidence = min(0.5 + 0.1 * (pos_count - neg_count), 0.8)
            score = 0.5
        elif neg_count > pos_count:
            label = '负面'
            confidence = min(0.5 + 0.1 * (neg_count - pos_count), 0.8)
            score = -0.5
        else:
            label = '中性'
            confidence = 0.5
            score = 0.0

        return SentimentScore(text=text[:200], label=label, confidence=round(confidence, 4), score=score)

    @staticmethod
    def _build_summary(scores: List[SentimentScore]) -> SentimentSummary:
        total = len(scores)
        if total == 0:
            return SentimentSummary(
                total_count=0, positive_count=0, negative_count=0, neutral_count=0,
                avg_score=0.0, positive_ratio=0.0, negative_ratio=0.0,
            )

        positive = sum(1 for s in scores if s.score > 0.2)
        negative = sum(1 for s in scores if s.score < -0.2)
        neutral = total - positive - negative
        avg = sum(s.score for s in scores) / total

        return SentimentSummary(
            total_count=total,
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            avg_score=round(avg, 4),
            positive_ratio=round(positive / total, 4),
            negative_ratio=round(negative / total, 4),
            scores=scores,
        )


# ========== 情感历史缓存（用于计算百分位）==========

class SentimentHistory:
    """
    情感历史缓存 — 记录每日情感分值，用于计算当前情感在历史中的百分位。

    数据存储: data/sentiment_history/{symbol}.json
    """
    HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sentiment_history')

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._ensure_dir()
        self.history = self._load()

    def _ensure_dir(self):
        os.makedirs(self.HISTORY_DIR, exist_ok=True)

    def _path(self) -> str:
        return os.path.join(self.HISTORY_DIR, f'{self.symbol}.json')

    def _load(self) -> list:
        try:
            if os.path.exists(self._path()):
                with open(self._path(), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('history', [])
        except Exception as e:
            logger.debug(f"情感历史加载失败 [{self.symbol}]: {e}")
        return []

    def _save(self, history: list):
        try:
            with open(self._path(), 'w', encoding='utf-8') as f:
                json.dump({'symbol': self.symbol, 'history': history}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"情感历史保存失败 [{self.symbol}]: {e}")

    def record(self, date_str: str, news_avg: float = 0.0, guba_avg: float = 0.0,
               news_count: int = 0, guba_count: int = 0):
        """记录当日情感数据，同日期覆盖"""
        # 移除同日期旧记录
        self.history = [h for h in self.history if h.get('date') != date_str]
        self.history.append({
            'date': date_str,
            'news_avg': news_avg,
            'guba_avg': guba_avg,
            'news_count': news_count,
            'guba_count': guba_count,
        })
        # 仅保留最近365天
        self.history = sorted(self.history, key=lambda x: x['date'])[-365:]
        self._save(self.history)

    def get_percentile(self, current_news_avg: float = 0.0, current_guba_avg: float = 0.0) -> dict:
        """计算当前情感在历史中的百分位"""
        if len(self.history) < 5:
            return {'news_percentile': None, 'guba_percentile': None, 'total_days': len(self.history)}

        news_scores = sorted([h.get('news_avg', 0) for h in self.history if h.get('news_avg') is not None])
        guba_scores = sorted([h.get('guba_avg', 0) for h in self.history if h.get('guba_avg') is not None])

        def _pct(val, arr):
            if not arr:
                return None
            return round(sum(1 for v in arr if v <= val) / len(arr) * 100, 1)

        return {
            'news_percentile': _pct(current_news_avg, news_scores),
            'guba_percentile': _pct(current_guba_avg, guba_scores),
            'total_days': len(self.history),
        }
