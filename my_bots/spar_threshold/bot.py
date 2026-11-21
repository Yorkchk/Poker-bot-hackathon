"""Sparring partner "threshold" (test opponent, NOT a tournament entry).

One of the practice bots vf was tested against (repo: poker_bot/baseline_bots.py,
exam_bots.py, versions/v1.py), bundled by tools/make_submission.py with the same
game_state adapter as vf. Kept in my_bots/ as allowed by the kit's README.
"""

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field


# ============================================================================
# poker_bot/bot.py: game state and actions
# ============================================================================
FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


@dataclass
class Action:
    kind: str            # fold / check / call / raise
    amount: int = 0      # for raise: the TOTAL we bet this street ("raise to")

    def __str__(self):
        return f"{self.kind} {self.amount}" if self.kind == RAISE else self.kind


@dataclass
class GameState:
    hole_cards: list         # our 2 cards (ints, see cards.py)
    board: list              # 0, 3, 4 or 5 cards
    pot: int                 # all chips in the middle, including this street's bets
    to_call: int             # chips we must add to call (already capped at our stack)
    stack: int               # chips we still have behind
    my_bet: int              # what we already put in this street
    current_bet: int         # highest bet this street
    min_raise_to: int | None # smallest legal "raise to", None if raising is impossible
    max_raise_to: int        # our all-in "raise to"
    big_blind: int
    num_opponents: int       # opponents still in the hand (not folded)
    num_players: int = 2     # players dealt into the hand
    street: str = "preflop"  # preflop / flop / turn / river
    hand_id: int | None = None   # game's hand number, if they give one
    my_id: str | None = None     # our player id
    players: list = field(default_factory=list)  # [{"id", "stack", "bet", "folded"}]
    actions: list = field(default_factory=list)  # whole hand so far, oldest first:


def legal_kinds(state):
    if state.to_call == 0:
        kinds = [CHECK]
    else:
        kinds = [FOLD, CALL]
    if state.min_raise_to is not None and state.max_raise_to > state.current_bet:
        kinds.append(RAISE)
    return kinds


def make_legal(state, action):
    """Turn any action into a legal one (clamp raises, swap fold<->check...)."""
    kinds = legal_kinds(state)
    if action.kind == RAISE:
        if RAISE not in kinds:
            return Action(CALL) if CALL in kinds else Action(CHECK)
        amount = max(state.min_raise_to, min(int(action.amount), state.max_raise_to))
        return Action(RAISE, amount)
    if action.kind == FOLD and CHECK in kinds:
        return Action(CHECK)          # never fold when checking is free
    if action.kind == CHECK and CHECK not in kinds:
        return Action(FOLD)
    if action.kind == CALL and CALL not in kinds:
        return Action(CHECK)
    return action


# ============================================================================
# cards.py
# ============================================================================
"""Card representation.

A card is an int 0..51:  card = rank * 4 + suit
  rank: 0 = '2', 1 = '3', ..., 8 = 'T', 9 = 'J', 10 = 'Q', 11 = 'K', 12 = 'A'
  suit: 0 = 'c', 1 = 'd', 2 = 'h', 3 = 's'
Ints keep the evaluator and the Monte Carlo loop fast.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"

FULL_DECK = list(range(52))


def card(text):
    """'As' -> int. Accepts upper/lower case, and '10' for ten."""
    text = text.strip()
    if text[:2] == "10":
        text = "T" + text[2:]
    rank = RANKS.index(text[0].upper())
    suit = SUITS.index(text[1].lower())
    return rank * 4 + suit


def cards(text):
    """'As Kd 7h' (spaces optional: 'AsKd7h') -> list of ints."""
    text = text.replace(" ", "").replace(",", "")
    return [card(text[i:i + 2]) for i in range(0, len(text), 2)]


def rank_of(c):
    return c >> 2


def suit_of(c):
    return c & 3


def to_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def hand_str(cs):
    return " ".join(to_str(c) for c in cs)


# ============================================================================
# evaluator.py
# ============================================================================
"""Hand evaluator: give it 5 to 7 cards, get back one int. Higher int = better hand.

Score layout:  category << 20 | five kicker ranks packed 4 bits each.
Comparing two hands is then just `a > b`, `a == b` means a split pot.

Instead of trying all 21 five-card subsets of 7 cards, we look at the
cards once (count ranks, collect ranks per suit) and read the best hand
off those counts. Lookup tables built at import make straights and
flushes a single list access.
"""


HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = [
    "High card", "Pair", "Two pair", "Three of a kind", "Straight",
    "Flush", "Full house", "Four of a kind", "Straight flush",
]


def _pack(category, ranks):
    value = 0
    for r in ranks:
        value = (value << 4) | r
    value <<= 4 * (5 - len(ranks))
    return (category << 20) | value


def _build_tables():
    popcount = [bin(m).count("1") for m in range(8192)]
    straight_high = [-1] * 8192
    top5 = [0] * 8192
    wheel = (1 << 12) | 0b1111  # A 2 3 4 5
    for mask in range(8192):
        for high in range(12, 3, -1):
            window = 0b11111 << (high - 4)
            if mask & window == window:
                straight_high[mask] = high
                break
        else:
            if mask & wheel == wheel:
                straight_high[mask] = 3  # five-high straight
        ranks = [r for r in range(12, -1, -1) if mask >> r & 1][:5]
        top5[mask] = _pack(0, ranks)  # kicker bits only
    return popcount, straight_high, top5


_POPCOUNT, _STRAIGHT_HIGH, _TOP5 = _build_tables()


def evaluate(cards):
    """Best 5-card hand score out of 5, 6 or 7 cards (ints from cards.py)."""
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] |= 1 << r

    # Flush / straight flush. With at most 7 cards a flush can never
    # coexist with quads or a full house, so checking it first is safe.
    for suit_mask in suits:
        if _POPCOUNT[suit_mask] >= 5:
            high = _STRAIGHT_HIGH[suit_mask]
            if high >= 0:
                return (STRAIGHT_FLUSH << 20) | (high << 16)
            return (FLUSH << 20) | _TOP5[suit_mask]

    quads, trips, pairs, singles = [], [], [], []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 1:
            singles.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 4:
            quads.append(r)

    if quads:
        q = quads[0]
        kicker = max([r for r in range(13) if counts[r] and r != q])
        return _pack(QUADS, [q, kicker])

    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        p = max(trips[1:] + pairs)
        return _pack(FULL_HOUSE, [t, p])

    high = _STRAIGHT_HIGH[suits[0] | suits[1] | suits[2] | suits[3]]
    if high >= 0:
        return (STRAIGHT << 20) | (high << 16)

    if trips:
        return _pack(TRIPS, [trips[0]] + singles[:2])

    if len(pairs) >= 2:
        kicker = max(pairs[2:] + singles[:1])
        return _pack(TWO_PAIR, [pairs[0], pairs[1], kicker])

    if pairs:
        return _pack(PAIR, [pairs[0]] + singles[:3])

    return _pack(HIGH_CARD, singles[:5])


def category(score):
    return score >> 20


def describe(score):
    """Human readable, e.g. 'Full house (K over 7)'. For logs/debugging."""
    cat = score >> 20
    ranks = [(score >> shift) & 0xF for shift in (16, 12, 8, 4, 0)]
    name = CATEGORY_NAMES[cat]
    if cat in (STRAIGHT, STRAIGHT_FLUSH):
        return f"{name}, {RANKS[ranks[0]]} high"
    if cat == FULL_HOUSE:
        return f"{name} ({RANKS[ranks[0]]} over {RANKS[ranks[1]]})"
    return f"{name} ({' '.join(RANKS[r] for r in ranks if r or cat == HIGH_CARD)})"


# ============================================================================
# preflop_table.py
# ============================================================================
"""Generated by tools/make_preflop_table.py: win chance of each starting hand
vs one random hand (12000 trials each). "AKs" = suited, "AKo" = offsuit."""

EQUITY_VS_RANDOM = {
    "AA": 0.8529,
    "KK": 0.8198,
    "QQ": 0.8032,
    "JJ": 0.775,
    "TT": 0.7464,
    "99": 0.7229,
    "88": 0.6912,
    "AQs": 0.6672,
    "AKs": 0.6668,
    "77": 0.6615,
    "AKo": 0.6574,
    "ATs": 0.6524,
    "AJs": 0.6486,
    "AQo": 0.6395,
    "AJo": 0.6356,
    "KQs": 0.635,
    "66": 0.6315,
    "ATo": 0.63,
    "A9s": 0.6262,
    "KJs": 0.624,
    "A8s": 0.6208,
    "KTs": 0.6201,
    "KQo": 0.61,
    "A7s": 0.6088,
    "KJo": 0.6077,
    "A9o": 0.6062,
    "QJs": 0.6033,
    "A6s": 0.603,
    "QTs": 0.6025,
    "55": 0.6022,
    "K9s": 0.6011,
    "A8o": 0.6001,
    "KTo": 0.5963,
    "A5s": 0.5874,
    "K8s": 0.5852,
    "A4s": 0.5842,
    "A7o": 0.5841,
    "K9o": 0.5829,
    "JTs": 0.5829,
    "K7s": 0.5821,
    "QJo": 0.5798,
    "A3s": 0.5794,
    "Q9s": 0.5782,
    "A5o": 0.578,
    "44": 0.5745,
    "A6o": 0.5737,
    "QTo": 0.5737,
    "A2s": 0.5736,
    "A4o": 0.5659,
    "Q9o": 0.5653,
    "K6s": 0.5634,
    "K5s": 0.5608,
    "A3o": 0.5589,
    "K8o": 0.5585,
    "Q8s": 0.5572,
    "J9s": 0.5572,
    "K7o": 0.5537,
    "K4s": 0.5496,
    "J8s": 0.5489,
    "K5o": 0.5449,
    "A2o": 0.5442,
    "T9s": 0.5442,
    "K6o": 0.5436,
    "JTo": 0.5435,
    "Q7s": 0.5432,
    "33": 0.5424,
    "K3s": 0.5399,
    "Q8o": 0.5349,
    "K2s": 0.5324,
    "Q6s": 0.5297,
    "J9o": 0.5292,
    "J7s": 0.529,
    "Q7o": 0.5232,
    "Q5s": 0.5227,
    "T9o": 0.522,
    "T8s": 0.5216,
    "J8o": 0.521,
    "K4o": 0.517,
    "Q4s": 0.516,
    "Q6o": 0.5149,
    "98s": 0.513,
    "K3o": 0.5128,
    "T7s": 0.5117,
    "K2o": 0.507,
    "Q5o": 0.5063,
    "J6s": 0.5063,
    "T8o": 0.5034,
    "J5s": 0.5018,
    "Q2s": 0.5015,
    "Q3s": 0.4997,
    "22": 0.4978,
    "J7o": 0.4952,
    "J4s": 0.492,
    "Q3o": 0.4861,
    "98o": 0.4851,
    "87s": 0.4851,
    "Q2o": 0.4843,
    "Q4o": 0.4833,
    "J3s": 0.483,
    "T7o": 0.483,
    "97s": 0.4823,
    "T6s": 0.4793,
    "96s": 0.4751,
    "J6o": 0.474,
    "T5s": 0.471,
    "J2s": 0.4704,
    "J4o": 0.4677,
    "J5o": 0.4667,
    "97o": 0.4657,
    "86s": 0.4612,
    "T3s": 0.4608,
    "T4s": 0.4604,
    "T6o": 0.46,
    "95s": 0.4583,
    "76s": 0.4575,
    "J3o": 0.4531,
    "T2s": 0.4508,
    "85s": 0.4474,
    "J2o": 0.4443,
    "96o": 0.442,
    "87o": 0.442,
    "T5o": 0.4406,
    "94s": 0.4397,
    "75s": 0.4387,
    "93s": 0.438,
    "T4o": 0.4365,
    "86o": 0.4352,
    "95o": 0.4343,
    "65s": 0.4305,
    "T3o": 0.4243,
    "84s": 0.42,
    "54s": 0.42,
    "85o": 0.4182,
    "76o": 0.4178,
    "T2o": 0.4164,
    "74s": 0.4163,
    "92s": 0.4155,
    "94o": 0.4148,
    "64s": 0.4093,
    "82s": 0.4073,
    "83s": 0.4067,
    "93o": 0.404,
    "73s": 0.404,
    "75o": 0.4035,
    "65o": 0.3986,
    "84o": 0.3932,
    "63s": 0.3923,
    "53s": 0.391,
    "43s": 0.3896,
    "74o": 0.3881,
    "64o": 0.3859,
    "72s": 0.3852,
    "54o": 0.3851,
    "92o": 0.385,
    "62s": 0.3777,
    "83o": 0.3762,
    "52s": 0.3756,
    "82o": 0.3744,
    "73o": 0.3686,
    "42s": 0.3635,
    "53o": 0.3629,
    "63o": 0.3586,
    "32s": 0.3573,
    "43o": 0.3525,
    "52o": 0.3519,
    "62o": 0.3452,
    "72o": 0.342,
    "32o": 0.3233,
    "42o": 0.3185,
}


# ============================================================================
# equity.py
# ============================================================================
"""Monte Carlo equity: chance of winning by playing the hand out at random many times.

Each trial: give every opponent 2 random cards, finish the board at random,
see who wins. Equity = (wins + share of split pots) / trials.

For now opponents hold *random* cards. The add-ons (hand ranges) will
plug in here later by sampling likely hands instead.
"""




def monte_carlo_equity(hole, board, n_opponents=1, iterations=2000,
                       time_budget=None, rng=None, ranges=None):
    """Return (equity between 0 and 1, number of trials actually run).

    hole: our 2 cards, board: 0-5 cards (ints from cards.py).
    Stops after `iterations` trials or `time_budget` seconds, whichever first.
    ranges: optional list with one entry per opponent: a list of 1,326 weights
            (see ranges.py) or None for "random cards". Without it, every
            opponent holds random cards (the original behaviour).
    """
    if n_opponents < 1:
        return 1.0, 0
    rng = rng or random
    if ranges is not None and any(r is not None for r in ranges):
        return _weighted_equity(hole, board, n_opponents, iterations, time_budget, rng, ranges)
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    board_missing = 5 - len(board)
    need = 2 * n_opponents + board_missing
    hole = list(hole)
    board = list(board)
    sample = rng.sample

    deadline = time.perf_counter() + time_budget if time_budget else None
    total = 0.0
    trials = 0
    while trials < iterations:
        drawn = sample(deck, need)
        full_board = board + drawn[2 * n_opponents:]
        mine = evaluate(hole + full_board)

        best_opp = -1
        ties = 0
        for i in range(n_opponents):
            s = evaluate(drawn[2 * i:2 * i + 2] + full_board)
            if s > best_opp:
                best_opp, ties = s, 1
            elif s == best_opp:
                ties += 1

        if mine > best_opp:
            total += 1.0
        elif mine == best_opp:
            total += 1.0 / (ties + 1)

        trials += 1
        if deadline and trials % 64 == 0 and time.perf_counter() > deadline:
            break

    return total / trials, trials


def _weighted_equity(hole, board, n_opponents, iterations, time_budget, rng, ranges):
    """Same as above, but each opponent's 2 cards are drawn according to weights."""
    from bisect import bisect_right
    from itertools import accumulate


    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    board_missing = 5 - len(board)
    hole = list(hole)
    board = list(board)

    ranges = list(ranges) + [None] * (n_opponents - len(ranges))
    cums = []
    for r in ranges[:n_opponents]:
        if r is None:
            r = [1.0] * len(COMBOS)
        w = [0.0 if (a in dead or b in dead) else max(0.0, x) for (a, b), x in zip(COMBOS, r)]
        if sum(w) <= 0:
            w = [0.0 if (a in dead or b in dead) else 1.0 for a, b in COMBOS]
        cums.append(list(accumulate(w)))

    random_ = rng.random
    sample = rng.sample
    deadline = time.perf_counter() + time_budget if time_budget else None
    total = 0.0
    trials = 0
    attempts = 0
    while trials < iterations and attempts < 3 * iterations:
        attempts += 1
        used = set()
        opp_hands = []
        ok = True
        for cum in cums:
            top = cum[-1]
            for _ in range(30):
                a, b = COMBOS[bisect_right(cum, random_() * top)]
                if a not in used and b not in used:
                    break
            else:
                ok = False
                break
            used.add(a)
            used.add(b)
            opp_hands.append([a, b])
        if not ok:
            continue             # extremely rare card clash: redraw
        if board_missing:
            extra = [c for c in sample(deck, min(len(deck), board_missing + 2 * n_opponents))
                     if c not in used][:board_missing]
            full_board = board + extra
        else:
            full_board = board
        mine = evaluate(hole + full_board)
        best_opp, ties = -1, 0
        for h in opp_hands:
            s = evaluate(h + full_board)
            if s > best_opp:
                best_opp, ties = s, 1
            elif s == best_opp:
                ties += 1
        if mine > best_opp:
            total += 1.0
        elif mine == best_opp:
            total += 1.0 / (ties + 1)
        trials += 1
        if deadline and trials % 64 == 0 and time.perf_counter() > deadline:
            break

    if trials == 0:
        return monte_carlo_equity(hole, board, n_opponents, iterations, time_budget, rng)
    return total / trials, trials


# ============================================================================
# replay.py
# ============================================================================
"""Replay a hand's action list and describe each action in context.

The game gives us a plain list of actions. Replaying it from the start tells
us, for every action: the pot at that moment, whether the player was facing
a bet, how many raises came before it on that street, and how big a raise
was compared to the pot. Profiles (step 3) and hand ranges (step 4) both use this.

Accepted action names: small_blind, big_blind, fold, check, call, bet, raise, all_in.
`amount` = the player's total bet on this street after the action.
"""


BLINDS = ("small_blind", "big_blind", "blind", "ante")


def replay(actions):
    contrib = defaultdict(int)        # chips each player put in this hand
    street_bet = defaultdict(int)     # chips each player put in this street
    street = None
    current_bet = 0
    raises = 0
    for a in actions:
        st = a.get("street", "preflop")
        if st != street:
            street, current_bet, raises = st, 0, 0
            street_bet = defaultdict(int)
        p = a.get("player")
        kind = str(a.get("action", "")).lower()
        amount = int(a.get("amount") or 0)
        pot_before = sum(contrib.values())
        to_call = max(0, current_bet - street_bet[p])

        if kind in ("all_in", "allin"):
            kind = "raise" if amount > current_bet else "call"
        if kind == "bet":
            kind = "raise"

        ctx = {
            "player": p, "street": st, "action": kind, "amount": amount,
            "pot_before": pot_before, "to_call": to_call, "facing_bet": to_call > 0,
            "raises_before": raises, "voluntary": kind not in BLINDS, "size": None,
        }

        if kind in BLINDS or kind in ("raise", "call"):
            new_total = amount
            if kind == "call" and amount <= street_bet[p]:
                new_total = current_bet       # format gave no amount: assume a full call
            if kind == "raise":
                ctx["size"] = (amount - current_bet) / max(1, pot_before + to_call)
                raises += 1
            added = max(0, new_total - street_bet[p])
            street_bet[p] += added
            contrib[p] += added
            current_bet = max(current_bet, street_bet[p])
        yield ctx


def raises_this_street(actions, street):
    """Number of raises (bets count as raises) made so far on `street`."""
    n = 0
    for ctx in replay(actions):
        if ctx["street"] == street and ctx["action"] == "raise":
            n += 1
    return n


def size_bucket(size):
    """Group a raise size (fraction of the pot) into a readable bucket."""
    if size is None:
        return None
    for limit, name in ((0.4, "1/3"), (0.6, "1/2"), (0.85, "3/4"), (1.2, "pot")):
        if size < limit:
            return name
    return "over"


# ============================================================================
# ranges.py
# ============================================================================
"""Step 4: what cards does each opponent probably hold?

For every opponent we keep a weight for each of the 1,326 possible two-card
hands (1.0 = fully believable, close to 0 = very unlikely). We replay the
hand's actions and, after each of their actions, shrink the hands that don't
fit it. The Monte Carlo simulation then deals them cards according to the
weights instead of uniformly at random.

How an action changes the weights:
  * every hand gets a position q between 0 (worst) and 1 (best) inside the
    opponent's current range: before the flop from the starting-hand table,
    after the flop from its real strength on the board at that moment
  * "they bet/raise with their top f" keeps hands with q above 1 - f;
    f comes from their profile (step 3) and shrinks for bigger bets
  * calls keep the middle-to-top, checks shrink the very top
  * edges are soft and weights never drop below a floor: bluffs and odd
    plays stay possible
"""



COMBOS = [(a, b) for a in range(52) for b in range(a + 1, 52)]   # 1,326 hands
SOFTNESS = 0.06       # how gradual the edge of "top f%" is
FLOOR = 0.03          # a weight multiplier never goes below this


def _hand_type(a, b):
    ra, rb = a >> 2, b >> 2
    hi, lo = max(ra, rb), min(ra, rb)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if (a & 3) == (b & 3) else "o")


def _preflop_positions():
    """q for every combo: share of all combos that are weaker (strongest ~1.0)."""
    types = sorted(EQUITY_VS_RANDOM, key=EQUITY_VS_RANDOM.get)   # weakest first
    size = {t: (6 if len(t) == 2 else 4 if t.endswith("s") else 12) for t in types}
    q_of_type, below = {}, 0
    for t in types:
        q_of_type[t] = (below + size[t] / 2) / 1326
        below += size[t]
    return [q_of_type[_hand_type(a, b)] for a, b in COMBOS]


PREFLOP_Q = _preflop_positions()


def _sigmoid(x):
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _top(q, f, floor):
    """Multiplier for 'they do this with their top f': ~1 above 1-f, ~floor below."""
    f = min(0.95, max(0.02, f))
    return floor + (1 - floor) * _sigmoid((q - (1 - f)) / SOFTNESS)


def _size_factor(size):
    """Bigger bets -> narrower, stronger range."""
    if size is None:
        return 1.0
    if size < 0.4:
        return 1.3
    if size < 0.85:
        return 1.0
    if size < 1.2:
        return 0.75
    return 0.55


class RangeBuilder:
    def __init__(self):
        self._board_cache = {}   # board tuple -> (strengths, order weakest->strongest)

    def _board_strengths(self, board):
        key = tuple(board)
        if key not in self._board_cache:
            if len(self._board_cache) > 64:
                self._board_cache.clear()
            used = set(board)
            strengths = [evaluate([a, b] + list(board)) if a not in used and b not in used else -1
                         for a, b in COMBOS]
            order = sorted(range(len(COMBOS)), key=strengths.__getitem__)
            self._board_cache[key] = (strengths, order)
        return self._board_cache[key]

    def _positions(self, weights, board):
        """q for each combo inside this opponent's weighted range on `board`."""
        strengths, order = self._board_strengths(board)
        total = sum(weights[i] for i in order if strengths[i] >= 0) or 1.0
        q = [0.0] * len(COMBOS)
        cum = 0.0
        i = 0
        n = len(order)
        while i < n and strengths[order[i]] < 0:   # impossible hands (use a board card)
            i += 1
        while i < n:                       # equal strengths share one position
            j = i
            group_w = 0.0
            while j < n and strengths[order[j]] == strengths[order[i]]:
                group_w += weights[order[j]]
                j += 1
            pos = (cum + group_w / 2) / total
            for k in range(i, j):
                q[order[k]] = pos
            cum += group_w
            i = j
        return q

    def build(self, state, profiles, hats=None, maniac_trust=0.15):
        """Return {player: weights} for opponents still in the hand, or None.

        hats: {player: hat} from hats.py. For a 'maniac', bets and raises say
        little about its cards, so each action narrows its range only
        `maniac_trust` as much as usual (0 = treat its cards as random)."""
        hats = hats or {}
        if not state.actions or not state.players:
            return None
        me = state.my_id
        alive = [p["id"] for p in state.players if not p.get("folded") and p["id"] != me]
        weights = {p: [1.0] * len(COMBOS) for p in alive}
        board_by_street = {"preflop": [], "flop": state.board[:3],
                           "turn": state.board[:4], "river": state.board[:5]}

        for ctx in replay(state.actions):
            p = ctx["player"]
            if p not in weights or not ctx["voluntary"] or ctx["action"] == "fold":
                continue
            prof = profiles.get(p)
            w = weights[p]
            kind, street = ctx["action"], ctx["street"]
            if street == "preflop":
                q = PREFLOP_Q
                vpip, pfr, three = prof.get("vpip"), prof.get("pfr"), prof.get("threebet")
                if kind == "raise":
                    f = pfr if ctx["raises_before"] == 0 else three if ctx["raises_before"] == 1 else three / 2
                    if ctx["size"] is not None and ctx["size"] > 1.2:
                        f *= 0.6              # huge raise or all-in: even stronger
                    mult = lambda qi: _top(qi, f, FLOOR)
                elif kind == "call":
                    cont = vpip if ctx["raises_before"] == 0 else max(0.05, vpip - pfr)
                    re_raise = three if ctx["raises_before"] >= 1 else pfr
                    mult = lambda qi: _top(qi, cont, FLOOR) * (1 - 0.5 * _top(qi, re_raise, 0.0))
                elif kind == "check":
                    mult = lambda qi: 1 - 0.5 * _top(qi, pfr, 0.0)
                else:
                    continue
            else:
                board = board_by_street.get(street, state.board)
                if len(board) < 3:
                    continue
                q = self._positions(w, board)
                agg, ftb = prof.get("aggression"), prof.get("fold_to_bet")
                bluff_floor = FLOOR + 0.2 * max(0.0, agg - 0.5)   # aggressive bots bluff more
                if kind == "raise":
                    f = (agg if not ctx["facing_bet"] else agg * 0.4) * _size_factor(ctx["size"])
                    mult = lambda qi: _top(qi, f, bluff_floor)
                elif kind == "call":
                    cont = 1 - ftb
                    mult = lambda qi: _top(qi, cont, FLOOR) * (1 - 0.5 * _top(qi, agg * 0.4, 0.0))
                elif kind == "check":
                    mult = lambda qi: 1 - 0.6 * _top(qi, agg, 0.0)
                else:
                    continue
            trust = maniac_trust if hats.get(p) == "maniac" else 1.0
            for i in range(len(COMBOS)):
                if w[i] > 0:
                    w[i] *= 1 - trust * (1 - mult(q[i]))
        return weights


# ============================================================================
# versions/v1.py
# ============================================================================
"""v1 (frozen): hand evaluator + Monte Carlo vs random cards, nothing else.

Exact copy of the first draft's decision logic. Do not edit.
"""



CONFIG_V1 = {
    "iterations": 1500,
    "time_budget": 0.15,
    "raise_ratio": 1.3,
    "big_raise_ratio": 1.6,
    "raise_size": 0.75,
    "big_raise_size": 1.0,
    "call_margin": 0.03,
    "commit_fraction": 0.5,
}


class BotV1:
    def __init__(self, name="v1", config=None, rng=None, **_ignored):
        self.name = name
        self.config = dict(CONFIG_V1, **(config or {}))
        self.rng = rng or random.Random()

    def decide(self, state):
        try:
            return make_legal(state, self._decide(state))
        except Exception:
            return Action(CHECK) if state.to_call == 0 else Action(FOLD)

    def _decide(self, state):
        cfg = self.config
        n = max(1, state.num_opponents)
        equity, _ = monte_carlo_equity(
            state.hole_cards, state.board, n,
            iterations=cfg["iterations"], time_budget=cfg["time_budget"], rng=self.rng,
        )
        strength = equity * (n + 1)
        pot_odds = state.to_call / (state.pot + state.to_call) if state.to_call else 0.0
        can_raise = RAISE in legal_kinds(state)

        if can_raise and strength >= cfg["raise_ratio"]:
            size = cfg["big_raise_size"] if strength >= cfg["big_raise_ratio"] else cfg["raise_size"]
            pot_after_call = state.pot + state.to_call
            raise_to = state.current_bet + int(size * pot_after_call)
            raise_to = max(raise_to, state.min_raise_to)
            added = raise_to - state.my_bet
            if added >= cfg["commit_fraction"] * state.stack:
                raise_to = state.max_raise_to
            return Action(RAISE, raise_to)

        if state.to_call == 0:
            return Action(CHECK)
        if equity > pot_odds + cfg["call_margin"]:
            return Action(CALL)
        return Action(FOLD)


# ============================================================================
# baseline_bots.py
# ============================================================================
"""Simple practice opponents. Most hackathon bots will resemble one of these."""




class RandomBot:
    def __init__(self, name="random", rng=None):
        self.name = name
        self.rng = rng or random.Random()

    def decide(self, state):
        kind = self.rng.choice(legal_kinds(state))
        if kind == RAISE:
            hi = min(state.max_raise_to, state.min_raise_to + 3 * state.pot)
            return make_legal(state, Action(RAISE, self.rng.randint(state.min_raise_to, max(hi, state.min_raise_to))))
        return Action(kind)


class CallingStation:
    """Calls everything, never raises."""

    def __init__(self, name="call"):
        self.name = name

    def decide(self, state):
        return Action(CHECK) if state.to_call == 0 else Action(CALL)


class Maniac:
    """Raises the pot every time it can."""

    def __init__(self, name="raise"):
        self.name = name

    def decide(self, state):
        return make_legal(state, Action(RAISE, state.current_bet + state.pot + state.to_call))


class TightBot:
    """Plays only good starting hands (pairs 8+, two high cards), then bets
    made pairs or better and folds the rest to bets. No simulation."""

    def __init__(self, name="tight"):
        self.name = name

    def decide(self, state):
        a, b = sorted((rank_of(c) for c in state.hole_cards), reverse=True)
        if state.street == "preflop":
            premium = (a == b and a >= 6) or (a >= 10 and b >= 9)   # 88+, AJ+, KQ
            if premium:
                return make_legal(state, Action(RAISE, state.current_bet + state.pot + state.to_call))
            return make_legal(state, Action(FOLD))
        board_ranks = {rank_of(c) for c in state.board}
        made = a == b or a in board_ranks or b in board_ranks
        if made:
            return make_legal(state, Action(RAISE, state.current_bet + (state.pot + state.to_call) // 2))
        return make_legal(state, Action(FOLD))


class ThresholdBot:
    """The 'typical hackathon bot': Monte Carlo equity vs random cards with fixed,
    un-normalised cut-offs (ignores how many players are in), always bets half pot."""

    def __init__(self, name="threshold", rng=None, raise_at=0.6, iterations=400):
        self.name = name
        self.rng = rng or random.Random()
        self.raise_at = raise_at
        self.iterations = iterations

    def equity(self, state):
        return monte_carlo_equity(state.hole_cards, state.board, max(1, state.num_opponents),
                                  iterations=self.iterations, rng=self.rng)[0]

    def decide(self, state):
        eq = self.equity(state)
        pot_odds = state.to_call / (state.pot + state.to_call) if state.to_call else 0.0
        if eq >= self.raise_at:
            return make_legal(state, Action(RAISE, state.current_bet + (state.pot + state.to_call) // 2))
        if state.to_call == 0:
            return Action(CHECK)
        return Action(CALL) if eq > pot_odds else Action(FOLD)


class Bluffer(ThresholdBot):
    """Threshold bot that, when it would check or fold, sometimes bets the pot anyway."""

    def __init__(self, name="bluffer", rng=None, bluff_rate=0.25):
        super().__init__(name=name, rng=rng)
        self.bluff_rate = bluff_rate

    def decide(self, state):
        action = super().decide(state)
        if action.kind in (CHECK, FOLD) and self.rng.random() < self.bluff_rate:
            return make_legal(state, Action(RAISE, state.current_bet + state.pot + state.to_call))
        return action


class Shover(ThresholdBot):
    """Threshold bot that, once short (<= 15 big blinds), goes all-in before the
    flop with the top 40% of starting hands and folds the rest."""

    def __init__(self, name="shover", rng=None):
        super().__init__(name=name, rng=rng)

    def decide(self, state):
        if state.street == "preflop" and state.stack + state.my_bet <= 15 * state.big_blind:
            if PREFLOP_Q[_combo_index(*state.hole_cards)] >= 0.6:
                return make_legal(state, Action(RAISE, state.max_raise_to))
            return make_legal(state, Action(FOLD))
        return super().decide(state)


class ChartBot:
    """A 'decent' bot that thinks differently from the equity bots: a starting-hand
    chart before the flop, simple made-hand / draw rules after it, varied bet sizes
    and occasional bluffs. No simulation."""

    def __init__(self, name="chart", rng=None):
        self.name = name
        self.rng = rng or random.Random()

    def decide(self, state):
        if state.street == "preflop":
            return make_legal(state, self._preflop(state))
        return make_legal(state, self._postflop(state))

    def _preflop(self, s):
        q = PREFLOP_Q[_combo_index(*s.hole_cards)]
        n = s.num_players
        play = 0.35 if n <= 3 else 0.25 if n <= 6 else 0.18     # play the top X% of hands
        facing_raise = s.current_bet > s.big_blind
        if not facing_raise:
            if q >= 1 - play:
                return Action(RAISE, 3 * s.big_blind + (s.pot - 1.5 * s.big_blind) // 2)
            return Action(CHECK) if s.to_call == 0 else Action(FOLD)
        if q >= 1 - play / 5:                                   # premium: re-raise 3x
            return Action(RAISE, 3 * s.current_bet)
        if q >= 1 - play / 2:
            return Action(CALL)
        return Action(FOLD)

    def _postflop(self, s):
        hole_ranks = sorted((c >> 2 for c in s.hole_cards), reverse=True)
        board_ranks = [c >> 2 for c in s.board]
        cat = category(evaluate(list(s.hole_cards) + list(s.board)))
        pot_after_call = s.pot + s.to_call
        price = s.to_call / pot_after_call if s.to_call else 0.0
        size = self.rng.choice((0.5, 0.66, 0.75))

        top_pair_good = (cat == PAIR and hole_ranks[0] in board_ranks
                         and hole_ranks[0] == max(board_ranks) and hole_ranks[1] >= 8)
        overpair = cat == PAIR and hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > max(board_ranks)
        strong = cat >= TWO_PAIR and (set(hole_ranks) & set(board_ranks) or hole_ranks[0] == hole_ranks[1]
                                     or cat >= 4)
        medium = cat == PAIR and (set(hole_ranks) & set(board_ranks) or hole_ranks[0] == hole_ranks[1])
        draw = s.street != "river" and _has_draw(s.hole_cards, s.board)

        if strong or top_pair_good or overpair:
            if s.to_call and not strong:
                return Action(CALL) if price <= 0.4 else Action(FOLD)
            return Action(RAISE, s.current_bet + int(size * pot_after_call))
        if medium:
            if s.to_call:
                return Action(CALL) if price <= 0.3 else Action(FOLD)
            if s.street == "river" and self.rng.random() < 0.3:
                return Action(RAISE, int(0.5 * s.pot))
            return Action(CHECK)
        if draw:
            if s.to_call:
                return Action(CALL) if price <= 0.25 else Action(FOLD)
            if self.rng.random() < 0.3:
                return Action(RAISE, int(0.5 * s.pot))
            return Action(CHECK)
        if not s.to_call and s.street == "river" and self.rng.random() < 0.12:
            return Action(RAISE, int(0.66 * s.pot))             # occasional bluff
        return Action(CHECK) if s.to_call == 0 else Action(FOLD)


class Folder(ThresholdBot):
    """Over-cautious bot: plays like the threshold bot, but after the flop it folds
    to any bet unless its hand is very strong (win chance >= 70%)."""

    def __init__(self, name="folder", rng=None):
        super().__init__(name=name, rng=rng)

    def decide(self, state):
        if state.street != "preflop" and state.to_call > 0 and self.equity(state) < 0.7:
            return Action(FOLD)
        return super().decide(state)


class Switcher:
    """Changes style mid-match: plays like the always-raise maniac for its first
    `switch_at` hands, then like a careful decent bot (v1-careful). Tests whether
    v2 notices the change instead of paying off its new, real strong hands."""

    def __init__(self, name="switcher", rng=None, switch_at=300):
        self.name = name
        self.switch_at = switch_at
        self.maniac = Maniac(name=name)
        self.careful = BotV1(name=name, rng=rng or random.Random(),
                             config={"raise_ratio": 1.5, "big_raise_ratio": 1.8})
        self.hands = 0
        self.last_hand = None

    def decide(self, state):
        key = state.hand_id if state.hand_id is not None else id(state.hole_cards)
        if key != self.last_hand:
            self.last_hand = key
            self.hands += 1
        bot = self.maniac if self.hands <= self.switch_at else self.careful
        return bot.decide(state)


def _has_draw(hole, board):
    cards = list(hole) + list(board)
    for suit in range(4):
        suited = [c for c in cards if c & 3 == suit]
        if len(suited) == 4 and any(c & 3 == suit for c in hole):
            return True
    ranks = {c >> 2 for c in cards}
    if 12 in ranks:
        ranks.add(-1)                                           # ace plays low too
    hole_ranks = {c >> 2 for c in hole}
    for lo in range(-1, 9):
        window = set(range(lo, lo + 4))
        if window <= ranks and window & (hole_ranks | ({-1} if 12 in hole_ranks else set())):
            return True                                         # open-ended straight draw
    return False


class Mixer:
    """A careful team that added randomness: v1's decisions, but random bet sizes,
    occasional slow-plays of strong hands and occasional bluffs."""

    def __init__(self, name="mixer", rng=None):
        self.name = name
        self.rng = rng or random.Random()
        self.base = BotV1(name=name, rng=self.rng)

    def decide(self, state):
        action = self.base.decide(state)
        r = self.rng.random()
        if action.kind == RAISE and state.to_call == 0 and state.street != "preflop" and r < 0.15:
            return Action(CHECK)                                # slow-play
        if action.kind == RAISE and action.amount < state.max_raise_to:
            size = self.rng.uniform(0.5, 1.0)
            return make_legal(state, Action(RAISE, state.current_bet + int(size * (state.pot + state.to_call))))
        if action.kind == CHECK and state.street in ("turn", "river") and r < 0.12:
            return make_legal(state, Action(RAISE, int(0.6 * state.pot)))   # bluff
        return action


def _combo_index(a, b):
    a, b = min(a, b), max(a, b)
    return a * 52 - a * (a + 1) // 2 + (b - a - 1)


BOTS = {
    "random": RandomBot,
    "call": CallingStation,
    "raise": Maniac,
    "tight": TightBot,
    "threshold": ThresholdBot,
    "bluffer": Bluffer,
    "shover": Shover,
    "chart": ChartBot,
    "mixer": Mixer,
    "folder": Folder,
    "switcher": Switcher,
}


# ============================================================================
# exam_bots.py
# ============================================================================
"""Final-exam opponents: bots that v2's settings were NEVER tuned against.

Rule: never change v2's settings based on results against these bots. They exist
only to check that v2 still does well against opponents it hasn't 'seen', which
is the honest test for overfitting. If v2 needs changing, make the change based
on the regular practice bots, then re-check here.

  tag       tight-aggressive: top 18% of hands, raises 2.5x, c-bets the flop,
            bets 2/3 pot with good equity
  category  no simulation: bets by made-hand type (pair small, two pair+ big)
  lag       loose-aggressive but sane: top 40% of hands, bets often, barrels
            draws, occasional bluff-raise
"""




def _q(hole):
    a, b = sorted(hole)
    return PREFLOP_Q[COMBOS.index((a, b))]


def _raised_preflop_last(state, me):
    last = None
    for a in state.actions or []:
        if a.get("street") == "preflop" and a.get("action") in ("raise", "bet"):
            last = a.get("player")
    return last == me


class TagBot:
    def __init__(self, name="tag", rng=None):
        self.name = name
        self.rng = rng or random.Random()

    def decide(self, s):
        if s.street == "preflop":
            q = _q(s.hole_cards)
            if s.current_bet <= s.big_blind:
                if q >= 0.82:
                    return make_legal(s, Action(RAISE, int(2.5 * s.big_blind)))
                return make_legal(s, Action(CHECK if s.to_call == 0 else FOLD))
            if q >= 0.95:
                return make_legal(s, Action(RAISE, 3 * s.current_bet))
            if q >= 0.88 and s.to_call <= 0.15 * s.stack:
                return Action(CALL)
            return make_legal(s, Action(FOLD))
        eq = monte_carlo_equity(s.hole_cards, s.board, max(1, s.num_opponents), 400, rng=self.rng)[0]
        pot = s.pot + s.to_call
        if s.to_call == 0:
            cbet = s.street == "flop" and _raised_preflop_last(s, self.name) and self.rng.random() < 0.65
            if eq >= 0.6 or cbet:
                return make_legal(s, Action(RAISE, int(0.66 * s.pot)))
            return Action(CHECK)
        if eq >= 0.8:
            return make_legal(s, Action(RAISE, s.current_bet + int(0.8 * pot)))
        return Action(CALL) if eq > s.to_call / pot else make_legal(s, Action(FOLD))


class CategoryBot:
    def __init__(self, name="category", rng=None):
        self.name = name
        self.rng = rng or random.Random()

    def decide(self, s):
        ranks = sorted((c >> 2 for c in s.hole_cards), reverse=True)
        if s.street == "preflop":
            suited = (s.hole_cards[0] & 3) == (s.hole_cards[1] & 3)
            playable = ranks[0] == ranks[1] or ranks[0] == 12 or (suited and ranks[1] >= 8)
            if playable and s.to_call <= 3 * s.big_blind:
                if ranks[0] == ranks[1] and ranks[0] >= 9:
                    return make_legal(s, Action(RAISE, 3 * max(s.current_bet, s.big_blind)))
                return Action(CHECK if s.to_call == 0 else CALL)
            return make_legal(s, Action(CHECK if s.to_call == 0 else FOLD))
        cat = category(evaluate(list(s.hole_cards) + list(s.board)))
        board_cat = category(evaluate(list(s.board))) if len(s.board) == 5 else PAIR - 1
        own = cat > board_cat
        if cat >= TWO_PAIR and own:
            return make_legal(s, Action(RAISE, s.current_bet + s.pot + s.to_call))
        if cat == PAIR and own:
            if s.to_call == 0:
                return make_legal(s, Action(RAISE, int(s.pot / 3)))
            return Action(CALL) if s.to_call <= 0.5 * s.pot else make_legal(s, Action(FOLD))
        return make_legal(s, Action(CHECK if s.to_call == 0 else FOLD))


class LagBot:
    def __init__(self, name="lag", rng=None):
        self.name = name
        self.rng = rng or random.Random()

    def decide(self, s):
        if s.street == "preflop":
            q = _q(s.hole_cards)
            if q >= 0.60 and s.current_bet <= s.big_blind:
                return make_legal(s, Action(RAISE, 3 * s.big_blind))
            if q >= 0.85 and self.rng.random() < 0.5:
                return make_legal(s, Action(RAISE, 3 * s.current_bet))
            if q >= 0.60 and s.to_call <= 0.1 * s.stack:
                return Action(CHECK if s.to_call == 0 else CALL)
            return make_legal(s, Action(CHECK if s.to_call == 0 else FOLD))
        eq = monte_carlo_equity(s.hole_cards, s.board, max(1, s.num_opponents), 400, rng=self.rng)[0]
        pot = s.pot + s.to_call
        if s.to_call == 0:
            if eq >= 0.45 or self.rng.random() < 0.35:
                return make_legal(s, Action(RAISE, int(self.rng.uniform(0.5, 0.9) * s.pot)))
            return Action(CHECK)
        if eq >= 0.7 or (s.street != "river" and self.rng.random() < 0.08):
            return make_legal(s, Action(RAISE, s.current_bet + int(0.9 * pot)))
        return Action(CALL) if eq > s.to_call / pot - 0.03 else make_legal(s, Action(FOLD))


EXAM_BOTS = {"tag": TagBot, "category": CategoryBot, "lag": LagBot}

SPARRING = "threshold"


def make(me):
    rng = random.Random(random.getrandbits(64))
    if SPARRING == "v1":
        return BotV1(name=me, rng=rng)
    if SPARRING == "v1-careful":
        return BotV1(name=me, rng=rng, config={"raise_ratio": 1.5, "big_raise_ratio": 1.8})
    cls = EXAM_BOTS.get(SPARRING) or BOTS[SPARRING]
    try:
        return cls(name=me, rng=rng)
    except TypeError:
        return cls(name=me)


# ============================================================================
# Tournament adapter: organizers' game_state -> GameState, vf's Action -> reply
# ============================================================================
SUBMISSION_CONFIG = {"time_budget": 0.5}   # half a CPU core in the sandbox; limit is 2 s
BOT_CLASS = None

_BOT = None
_FED = set()          # hand numbers already fed to the profiles


def _street_totals(entries):
    """Organizers' log -> vf's action list. vf wants `amount` = the player's total bet
    on the street; in their log a call or a blind post gives the chips added, a raise or
    all-in the street total, and antes are dead money (skipped)."""
    out, street, bet = [], None, {}
    for e in entries:
        kind = str(e.get("action", "")).lower()
        st = e.get("street", "preflop")
        if kind in ("ante", "show") or st == "showdown":
            continue
        if st != street:
            street, bet = st, {}
        p = e.get("bot_id")
        amt = int(e.get("amount") or 0)
        if kind in ("small_blind", "big_blind", "call"):
            bet[p] = bet.get(p, 0) + amt
            amt = bet[p]
        elif kind in ("raise", "all_in"):
            bet[p] = max(bet.get(p, 0), amt)
            amt = bet[p]
        else:
            amt = 0
        out.append({"player": p, "street": st, "action": kind, "amount": amt})
    return out


def _to_state(g):
    me = g.get("your_bot_id")
    my_bet = int(g.get("your_bet_this_street") or 0)
    stack = int(g.get("your_stack") or 0)
    current = int(g.get("current_bet") or 0)
    owed = int(g.get("amount_owed") or 0)
    players = [{"id": p.get("bot_id"), "stack": int(p.get("stack") or 0),
                "bet": int(p.get("bet_this_street") or 0),
                "folded": bool(p.get("is_folded")) or p.get("state") == "folded"}
               for p in g.get("players") or [] if p.get("state") != "busted"]
    rivals = [p for p in players if not p["folded"] and p["id"] != me]
    max_to = my_bet + stack
    min_to = None
    if max_to > current and g.get("min_raise_to") is not None and any(p["stack"] > 0 for p in rivals):
        min_to = min(int(g["min_raise_to"]), max_to)
    return GameState(
        hole_cards=[card(c) for c in g["your_cards"]],
        board=[card(c) for c in g.get("community_cards") or []],
        pot=int(g.get("pot") or 0), to_call=min(owed, stack), stack=stack, my_bet=my_bet,
        current_bet=current, min_raise_to=min_to, max_raise_to=max_to,
        big_blind=int((g.get("blinds") or {}).get("bb") or 1),
        num_opponents=max(1, len(rivals)), num_players=len(players),
        street=g.get("street", "preflop"), hand_id=g.get("hand_id"), my_id=me,
        players=players, actions=_street_totals(g.get("action_log") or []))


def _feed_finished_hands(g, bot, me):
    """Every hand older than the current one is finished: give it to the profiles once.
    The log keeps the last 200 entries; a hand whose start scrolled out is skipped."""
    current = g.get("hand_num")
    by_hand = {}
    for e in g.get("match_action_log") or []:
        h = e.get("hand_num")
        if h is None or h == current or h in _FED:
            continue
        by_hand.setdefault(h, []).append(e)
    for h in sorted(by_hand):
        entries = by_hand[h]
        _FED.add(h)
        if not any(e.get("action") in ("small_blind", "big_blind") for e in entries):
            continue                                    # start of the hand cut off
        streets = [e.get("street") for e in entries if e.get("action") not in ("show", "ante")]
        board_len = max((("preflop", "flop", "turn", "river").index(s) for s in streets
                         if s in ("preflop", "flop", "turn", "river")), default=0)
        ids = []
        for e in entries:
            if e.get("bot_id") not in ids:
                ids.append(e.get("bot_id"))
        bot._learn({
            "hand_id": "log-%s" % h, "me": me, "players": [{"id": i} for i in ids],
            "board": ["?"] * (0 if board_len == 0 else board_len + 2),
            "actions": _street_totals(entries),
            "shown": {e.get("bot_id"): e.get("cards") for e in entries if e.get("action") == "show"},
            "result": None, "complete": False})         # no chip results in the log


def _reply(state, a):
    if a.kind == RAISE:
        if a.amount >= state.max_raise_to:
            return {"action": "all_in"}
        return {"action": "raise", "amount": int(a.amount)}
    if a.kind == CALL:
        return {"action": "call"}
    if a.kind == CHECK:
        return {"action": "check"}
    return {"action": "fold"}


def decide(game_state):
    """Called by the tournament for every decision (and once with {"type": "warmup"})."""
    global _BOT
    try:
        if game_state.get("type") == "warmup":
            return {"action": "check"}
        me = game_state.get("your_bot_id") or "me"
        if _BOT is None:
            _BOT = make(me)
        try:
            _feed_finished_hands(game_state, _BOT, me)
        except Exception:
            pass
        state = _to_state(game_state)
        return _reply(state, _BOT.decide(state))
    except Exception:
        return {"action": "check"} if game_state.get("can_check") else {"action": "fold"}
