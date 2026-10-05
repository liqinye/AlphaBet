# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classify Polymarket events by matching topical tags to domain buckets.

Market listing flags are separated before classification. The primary domain
has the most matching tags, with ties resolved by _PRIORITY; domains() returns
all matches. audit_vocabulary() reports unmapped tags and unclassified events.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

UNKNOWN = "unknown"

NON_TOPICAL = re.compile(
    r"^(?:"
    r"hide-from-new|recurring|weekly|monthly|daily|yearly|all|new|trending|featured|live"
    r"|neg-risk|parlays?|parlay|multi-strikes|game-specials|combos?|up-or-down|close"
    r"|mention-markets|tweets-markets|breaking-news|prediction-markets|kpis|math"
    r"|rewards[\w-]*|finance-rewards-\d+"
    # Match bare expiry dates while retaining tags such as march-3-primaries.
    r"|(?:january|february|march|april|may|june|july|august|september|october"
    r"|november|december)-\d{1,2}"
    r")$"
)

DOMAIN_TAGS: dict[str, set[str]] = {
    "crypto": {
        "crypto", "crypto-prices", "bitcoin", "ethereum", "solana", "xrp", "ripple",
        "microstrategy", "bnb", "binance", "aave", "dash", "cz", "ath", "token-launch",
        "hit-price", "paradex", "alpha-arena", "btm",
    },
    "sports": {
        "sports", "soccer", "premier-league", "EPL", "la-liga", "ucl", "uel", "bundesliga",
        "ligue-1", "primeira-liga", "champions-league", "copa-del-rey", "conmebol",
        "uef-qualifiers", "afcon", "africa-cup-of-nations", "mls", "football", "england",
        "nfl", "nfl-playoffs", "nfl-free-agency", "free-agency", "super-bowl", "superbowl",
        "super-bowl-lix-game-props", "super-bowl-lx", "big-game", "patriots", "seahawks",
        "thanksgiving-football", "half-time", "margin-of-victory", "mov",
        "nba", "nba-all-star", "nba-all-star-game", "all-star", "all-star-game", "lakers",
        "luka-doncic", "basketball", "euro-basket", "eurobasket", "cbb", "cfb", "lane-kiffin",
        "hockey", "iihf", "wjc", "tennis", "atp", "us-open", "miami-open", "mens-singles",
        "golf", "pga", "pga-tour", "masters", "augusta-national", "tiger", "tiger-woods",
        "hole-in-one", "f1", "formula1", "grand-prix", "f1-dutch-gp", "united-states-grand-prix",
        "boxing", "jake-paul", "jake-paul-vspt-joshua", "ufc", "grappling", "cricket",
        "cricket-test", "international-cricket", "olympics", "winter-games", "medals",
        "medal-count", "gold-medals", "climbing", "free-solo", "skyscraper-climb",
        "esports", "counter-strike", "counter-strike-2", "cs2", "iem-rio-2026", "ewcz",
        "dota", "dota-2", "lol", "lol-worlds-2025", "league-of-legends", "rl", "chess",
        "as", "aus", "bor", "acm", "open",
    },
    "politics": {
        "politics", "us-politics", "uptspt-poltics", "elections", "election",
        "global-elections", "world-elections", "world-elctions", "mayoral-elections",
        "senate-elections", "virginia-elections", "primaries", "march-3-primaries",
        "nov-4-elections", "us-presidential-election", "world-presidency", "referendum",
        "redistricting", "texas-redistricting", "trump", "trump-presidency", "trump-cabinet",
        "vance", "kushner", "democrats", "congress", "congress-expulsions", "expel",
        "house", "house-of-representatives", "senate", "courts", "approval", "approvals",
        "poty", "sotu", "state-of-union", "state-of-the-union", "executive-actions",
        "governance", "gov-shutdown", "shutdown", "swalwell", "mullin", "talarico",
        "tim-walz", "nyc-mayor", "zohran-mamdani", "mamdani", "new-york", "texas-senate",
        "epstein", "charlie-kirk", "greta-thunberg", "immigration", "ice", "h-1b",
        "homeland-security", "department-of-homeland-security", "dhs", "noem",
        "minnesota-fraud", "starmer", "modi", "orban", "fidesz", "tisza", "magyar",
        "hungarian", "hungary-election", "portugal-election", "portuguese-elections",
        "dutch-election", "dutch", "french-elections", "japan-election", "ireland-election",
        "irish", "bangladesh-election", "thailand-election", "slovenia-elections",
        "moldova-election", "bucharest-mayor", "powerball", "taxes", "nobel-peace-prize",
        "trump-machado", "maria-corina-machado", "reza-pahlavi",
    },
    "geopolitics": {
        "geopolitics", "world", "world-affairs", "foreign-policy", "diplomacy-ceasefire",
        "middle-east", "israel", "netanyahu", "israel-strike-iran", "israel-x-iran",
        "iran", "trump-iran", "iran-offensive-actions", "iranian-leadership-regime",
        "khamenei", "ali-khamenei", "gaza", "gaza-floatilla", "hamas", "hezbollah",
        "houthis", "houtis", "yemen", "syria", "lebanon", "qatar", "quatar",
        "ukraine", "ukraine-map", "ukraine-peace-deal", "russia", "russia-capture",
        "putin", "trump-putin", "zelenskyy", "trump-zelenskyy", "witkoff",
        "venezuela", "maduro", "cartel", "mexico-cartel-war", "el-mencho",
        "china", "trump-xi", "taiwan", "hong-kong", "north-korea", "south-korea",
        "india-pakistan", "thailand-cambodia", "somalia", "cuba", "greenland",
        "military", "military-action", "army", "dod", "soldiers", "airspace", "shoot",
        "nuclear", "nuclear-deal", "regional-spillover", "flee",
        "strait-of-hormuz", "hormuz", "hormoz", "tanker", "oil-ship", "ship", "shipping",
        "sea", "liquefied", "trade-war", "tariffs", "trade", "davos", "cop",
        "gustavo-petro", "gorton-and-denton",
    },
    "finance": {
        "finance", "financials", "earnings", "pre-market", "economy", "economic-policy",
        "equities", "stocks", "stock-prices", "sp-500", "s-and-p", "spx", "dji", "msci",
        "indicies", "futures", "derivatives", "cboe-volatility-index", "macro-indicators",
        "Global-Rates", "rate", "fed", "fed-rates", "fomc", "jerome-powell", "rba",
        "inflation", "cpi", "gdp", "jobs-report", "eurozone", "treasuries", "exchange-rate",
        "usd", "commodities", "gold", "silver", "comex", "comex-silver-futures", "oil",
        "natural-gas", "lng", "uranium", "real-estate", "acquisitions", "bank", "funding",
        "business", "companies", "msm",
        "margin", "waste-management", "air-travel", "travel", "traffic", "up-or-down",
        # Raw ticker tags on earnings and price markets count toward finance.
        "abnb", "acn", "amat", "amzn", "arcc", "arm", "asan", "baba", "bbwi", "bk",
        "bkng", "blk", "bmbl", "c", "ci", "crl", "crm", "crsp", "cvx", "dal", "dbx",
        "dd", "ddog", "dell", "dis", "efx", "fdx", "fis", "fsk", "ftdr", "gamb", "gap",
        "grab", "gs", "hesm", "hims", "hum", "ibkr", "intu", "itc", "ivz", "jblu",
        "jef", "jnj", "jpm", "lh", "lly", "lulu", "lw", "m", "ma", "mbwm", "mcd",
        "mcdonalds", "mdb", "mmm", "mnst", "mrna", "msm", "nestle", "nflx", "nvda",
        "okta", "oracle", "orcl", "pg", "pipr", "pltr", "pm", "powl", "psky", "pypl",
        "qcom", "rbrk", "rcl", "rddt", "sbux", "schw", "sndk", "snex", "sofi", "stt",
        "stx", "tsla", "tw", "uber", "unh", "ups", "wb", "wbd", "wfc", "wmt", "xom",
        "kitkat", "warner-bros", "alibaba", "ups",
    },
    "tech": {
        "tech", "big-tech", "ai", "artificial-intelligence", "openai", "gpt", "gpt-5",
        "sora", "anthropic", "claude", "claude-5", "gemini", "gemini-3", "grok", "xai",
        "deepseek", "apple", "app", "app-store", "google", "googl", "google-search",
        "aws", "amazon", "microsoft", "msft", "nvidia", "tesla", "elon-musk", "twitter",
        "threads", "internet", "iot", "self-driving", "downtime", "outage", "outages",
        "steam", "deft", "vsco", "tea-dating-advice", "TBPN", "bryan-johnson", "dont-die",
        "larry-ellison", "columbia",
    },
    "pop_culture": {
        "pop-culture", "games", "video-games", "game-awards", "movies", "box-office",
        "avatar", "netflix", "top-netflix", "tv", "reality-tv", "streamer", "nielsen",
        "viewers", "views", "media", "magazine", "music", "song", "songs", "album",
        "albums", "billboard", "charts", "hot-100", "spotify", "youtube", "mrbeast",
        "mafiathon", "awards", "celebrities", "celebrity", "taylor-swift", "drake",
        "badbunny", "justin-bieber", "kanye", "ye", "sabrina-carpenter", "david",
        "halftime-show", "best-of-2025", "2025-predictions", "christmas", "thanksgiving",
        "catholic", "rating", "mogpta",
    },
    "science_health_weather": {
        "science", "climate-science", "weather", "climate", "climate-weather",
        "global-warming", "global-temp", "daily-temperature", "highest-temperature",
        "precipitation", "snow", "snow-storm", "hurricanes", "natural-disaster",
        "natural-disasters", "earthquake", "earthquakes", "pandemics", "influenza",
        "flu", "drugs",
    },
    "crime_society": {
        "crime", "theft", "robbery", "heist", "strike", "labor", "education", "tsa",
        "nyt", "paris", "london", "los-angeles", "seattle", "minneapolis",
        "arizona", "california", "texas", "massachusetts", "massachussetts", "minnesota",
        "quebec", "canada", "brazil", "argentina", "chile", "colombia", "bolivia",
        "panama", "mexico", "japan", "india", "vietnam", "thailand", "france", "norway",
        "netherlands", "ireland", "romania", "bulgaria", "hungary", "moldova", "slovenia",
        "uk", "united-kingdom", "south-america", "africa", "world-elections-misc",
    },
}

# Break ties by subject specificity; the geography-heavy crime_society bucket is last.
_PRIORITY = (
    "crypto",
    "sports",
    "geopolitics",
    "politics",
    "finance",
    "tech",
    "pop_culture",
    "science_health_weather",
    "crime_society",
)

# Each tag must belong to exactly one domain.
_TAG_TO_DOMAIN: dict[str, str] = {}
for _dom, _tags in DOMAIN_TAGS.items():
    for _t in _tags:
        _prev = _TAG_TO_DOMAIN.setdefault(_t, _dom)
        if _prev != _dom:  # pragma: no cover - definition-time guard
            raise AssertionError(f"tag {_t!r} claimed by both {_prev!r} and {_dom!r}")

assert set(_PRIORITY) == set(DOMAIN_TAGS), "_PRIORITY must cover exactly DOMAIN_TAGS"

DOMAINS: tuple[str, ...] = _PRIORITY


def is_non_topical(tag: str) -> bool:
    """True for Polymarket listing-mechanics flags (`weekly`, `neg-risk`, `rewards-*`)."""
    return bool(NON_TOPICAL.match(tag))


def split_tags(tags: Iterable[str]) -> tuple[list[str], list[str]]:
    """Partition raw tags into `(topical, market_flags)`, order preserved."""
    topical, flags = [], []
    for t in tags or ():
        (flags if is_non_topical(t) else topical).append(t)
    return topical, flags


def domain_votes(tags: Iterable[str]) -> dict[str, int]:
    """`{domain: n_matching_tags}` over the topical tags only (empty if none match)."""
    votes: dict[str, int] = {}
    topical, _ = split_tags(tags)
    for t in topical:
        dom = _TAG_TO_DOMAIN.get(t)
        if dom:
            votes[dom] = votes.get(dom, 0) + 1
    return votes


def domains(tags: Iterable[str]) -> list[str]:
    """Return all matching domains in _PRIORITY order."""
    votes = domain_votes(tags)
    return [d for d in _PRIORITY if d in votes]


def classify(tags: Iterable[str]) -> str:
    """Return the domain with the most tag matches, or UNKNOWN; ties use _PRIORITY."""
    votes = domain_votes(tags)
    if not votes:
        return UNKNOWN
    best = max(votes.values())
    for dom in _PRIORITY:
        if votes.get(dom) == best:
            return dom
    return UNKNOWN  # pragma: no cover - unreachable given _PRIORITY covers DOMAIN_TAGS


def annotate(tags: Iterable[str]) -> dict[str, Any]:
    """Return domain labels, topical tags, and market flags for row metadata.

    Keep these beside metadata["question"]: question fields enter prompts
    and determine evaluation group hashes.
    """
    tags = list(tags or ())
    topical, flags = split_tags(tags)
    return {
        "domain": classify(tags),
        "domains": domains(tags),
        "tags": topical,
        "market_flags": flags,
    }


def audit_vocabulary(tag_lists: Iterable[Iterable[str]]) -> dict[str, Any]:
    """Count unclassified events and unmapped topical tags in a corpus."""
    unmapped: dict[str, int] = {}
    unresolved = n = 0
    per_domain: dict[str, int] = {}
    for tags in tag_lists:
        n += 1
        topical, _ = split_tags(tags)
        for t in topical:
            if t not in _TAG_TO_DOMAIN:
                unmapped[t] = unmapped.get(t, 0) + 1
        dom = classify(tags)
        per_domain[dom] = per_domain.get(dom, 0) + 1
        if dom == UNKNOWN:
            unresolved += 1
    return {
        "n_events": n,
        "n_unresolved": unresolved,
        "per_domain": dict(sorted(per_domain.items(), key=lambda kv: -kv[1])),
        "n_unmapped_tags": len(unmapped),
        "unmapped_tags": dict(sorted(unmapped.items(), key=lambda kv: -kv[1])),
    }
