# poker_cli.py
from dataclasses import dataclass
import random, sys, math, os
from collections import Counter, deque
from itertools import combinations

# ---- deps ----
# Import required libraries. Numpy is mandatory, others are optional.
try:
    import numpy as _np
except Exception:
    raise SystemExit("This build requires numpy. pip install numpy")
try:
    import pandas as pd
except Exception:
    pd = None  # if pandas not available, CSV export still works with builtin csv
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True   # Torch is optional; only needed for actor-critic agent
except Exception:
    TORCH_AVAILABLE = False

# ---- config ----
# Game + training hyperparameters
STARTING_STACK = 1000
SMALL_BLIND = 5
BIG_BLIND = 10
AI_NAME = "Daniel Negreanu"   # default AI name
MC_TRIALS = {"preflop": 600, "flop": 1200, "turn": 2000, "river": 0}  # Monte Carlo samples
LEARNING_RATE = 0.01
RL_BASELINE_MOMENTUM = 0.9
IMITATION_EPOCHS = 3
TRAIN_HANDS_PER_EPOCH = 300
VAL_SPLIT = 0.2
TORCH_HIDDEN = 128
TORCH_LR = 3e-3
TORCH_ENTROPY_BETA = 0.01
REPLAY_CAPACITY = 2000
REPLAY_BATCH = 64
SELFPLAY_EPISODES = 300
SAVE_HISTORY_CSV = "hand_history.csv"
SAVE_TOURNEY_CSV = "tourney_log.csv"

# ---- unicode suits ----
# Handle whether terminal supports Unicode (♥♦♣♠) or fall back to ASCII (h/d/c/s).
def _supports_unicode():
    enc = (sys.stdout.encoding or "").upper()
    return "UTF" in enc or "UTF" in (sys.getdefaultencoding() or "").upper()

USE_UNICODE_SUITS = _supports_unicode()
SUIT_CODES = ["S","H","D","C"]
SUIT_FULLNAME = {"S":"Spades","H":"Hearts","D":"Diamonds","C":"Clubs"}
SUIT_CHAR_UNICODE = {"S":"♠","H":"♥","D":"♦","C":"♣"}
SUIT_CHAR_ASCII = {"S":"s","H":"h","D":"d","C":"c"}

# ---- cards/deck/eval ----
# Define card ranks and poker hand strength categories.
RANKS = list(range(2,15))
HAND_RANK = {
    "high":0,"pair":1,"two_pair":2,"trips":3,"straight":4,
    "flush":5,"full_house":6,"quads":7,"straight_flush":8
}
HAND_RANK_NAMES = {v:k for k,v in HAND_RANK.items()}

@dataclass(frozen=True)
class Card:
    rank:int; suit:str
    # Full name string (e.g. "Ace of Spades")
    def __str__(self): return f"{self.rank_name()} of {SUIT_FULLNAME[self.suit]}"
    # Short name (e.g. "As", "K♥")
    def short(self):
        r={11:"J",12:"Q",13:"K",14:"A"}.get(self.rank,str(self.rank))
        s=SUIT_CHAR_UNICODE[self.suit] if USE_UNICODE_SUITS else SUIT_CHAR_ASCII[self.suit]
        return f"{r}{s}"
    # Rank label for display
    def rank_name(self): 
        return {11:"Jack",12:"Queen",13:"King",14:"Ace"}.get(self.rank, str(self.rank))

# Build a full 52-card deck
ALL_CARDS = [Card(r,s) for r in RANKS for s in SUIT_CODES]

class Deck:
    def __init__(self): 
        self.cards = ALL_CARDS.copy(); random.shuffle(self.cards)
    def deal(self,n=1): 
        out=self.cards[:n]; self.cards=self.cards[n:]; return out

# Hand evaluator: determine best 5-card hand out of 7 cards
def _best5_score(seven):
    ranks=[c.rank for c in seven]; suits=[c.suit for c in seven]
    rc=Counter(ranks); sc=Counter(suits)
    flush_suit=next((s for s,c in sc.items() if c>=5), None)
    flush_ranks=sorted([c.rank for c in seven if c.suit==flush_suit], reverse=True) if flush_suit else []
    # Detect straight high card (with Ace-low support)
    def straight_high(rs):
        rset=set(rs)
        if 14 in rset: rset.add(1)
        for hi in range(14,4,-1):
            if {hi,hi-1,hi-2,hi-3,hi-4}.issubset(rset): return 5 if hi==5 else hi
        return None
    straight_hi=straight_high(ranks)
    sf_hi=straight_high(flush_ranks) if flush_ranks else None
    # Hand rankings in descending order
    if sf_hi: return (HAND_RANK["straight_flush"], sf_hi)
    if 4 in rc.values():
        q=max(r for r,c in rc.items() if c==4); k=max(r for r,c in rc.items() if c!=4)
        return (HAND_RANK["quads"], q, k)
    trips=sorted([r for r,c in rc.items() if c==3], reverse=True)
    pairs=sorted([r for r,c in rc.items() if c==2], reverse=True)
    if trips and (len(trips)>1 or pairs):
        t=trips[0]; s=trips[1] if len(trips)>1 else pairs[0]
        return (HAND_RANK["full_house"], t, s)
    if flush_ranks: return (HAND_RANK["flush"], *flush_ranks[:5])
    if straight_hi: return (HAND_RANK["straight"], straight_hi)
    if trips:
        t=trips[0]; kickers=sorted([r for r in ranks if r!=t], reverse=True)[:2]
        return (HAND_RANK["trips"], t, *kickers)
    if len(pairs)>=2:
        p1,p2=pairs[:2]; hi,lo=max(p1,p2),min(p1,p2)
        k=max(r for r in ranks if r not in (p1,p2))
        return (HAND_RANK["two_pair"], hi, lo, k)
    if len(pairs)==1:
        p=pairs[0]; kickers=sorted([r for r in ranks if r!=p], reverse=True)[:3]
        return (HAND_RANK["pair"], p, *kickers)
    return (HAND_RANK["high"], *sorted(ranks, reverse=True)[:5])

class HandEvaluator:
    @staticmethod
    def evaluate(hole, board): 
        # Evaluate the strength of the hand (hole + board combined) using best 5-card score
        return _best5_score(hole+board)

def monte_carlo_equity(hole, board, opponents=1, trials=1000):
    # Estimate equity (probability of winning) via Monte Carlo simulation
    known=set(hole+board)  # cards already visible
    rem=[c for c in ALL_CARDS if c not in known]  # remaining deck
    need=5-len(board)  # number of board cards still to come

    # Exact calculation if river complete, one opponent, and trials <= 0
    if need==0 and opponents==1 and trials<=0:
        wins=ties=0; my=HandEvaluator.evaluate(hole,board)
        for h1,h2 in combinations(rem,2):  # try all opponent hole combos
            opp=HandEvaluator.evaluate([h1,h2],board)
            if my>opp: wins+=1
            elif my==opp: ties+=1
        total=len(rem)*(len(rem)-1)//2
        return (wins+0.5*ties)/max(total,1)

    # Monte Carlo sampling (approximate)
    wins=ties=0; draws=need+2*opponents; T=max(trials,1)
    for _ in range(T):
        picks=random.sample(rem, draws)  # random unknown cards
        b=board+picks[:need]; cur=need
        opp_scores=[]
        for _o in range(opponents):
            oh=picks[cur:cur+2]; cur+=2
            opp_scores.append(HandEvaluator.evaluate(oh,b))
        my=HandEvaluator.evaluate(hole,b)
        best=max([my]+opp_scores); cnt=([my]+opp_scores).count(best)
        if my==best and cnt==1: wins+=1
        elif my==best: ties+=1
    return (wins+0.5*ties)/T

# ---- advice helpers ----
def fmt_cards(cards): 
    # Format a list of cards into a short string (or dash if empty)
    return " ".join(c.short() for c in cards) if cards else "—"

def classify_pair_detail(hole, board):
    # Refined pair classification (overpair, top pair, second pair)
    if not board: 
        return "One Pair" if len(hole)==2 and hole[0].rank==hole[1].rank else "High Card"
    br=sorted([c.rank for c in board], reverse=True); hr=[c.rank for c in hole]
    if len(hole)==2 and hole[0].rank==hole[1].rank and hole[0].rank>max(br): return "Overpair"
    if any(r==br[0] for r in hr): return "Top Pair"
    if len(br)>1 and any(r==br[1] for r in hr): return "Second Pair"
    return "One Pair"

def classify_hand(hole,board):
    # Return hand category (pair, flush, straight, etc.)
    cat=HAND_RANK_NAMES[HandEvaluator.evaluate(hole,board)[0]]
    return classify_pair_detail(hole,board) if cat=="pair" else cat.replace("_"," ").title()

def has_flush_draw(hole,board):
    # Detect if hand currently has a flush draw (4 cards of same suit)
    sc=Counter([c.suit for c in hole+board])
    for s,c in sc.items():
        if c==4 and any(h.suit==s for h in hole): return True,s
    return False,None

def straight_draw_type(hole,board):
    # Detect straight draw type: OESD (open-ended) or Gutshot (inside)
    rset=set(c.rank for c in hole+board)
    if 14 in rset: rset=set(rset)|{1}  # Ace can act as 1
    for x in range(1,11):
        if {x,x+1,x+2,x+3}.issubset(rset) and ((x+4) not in rset): return "OESD"
        if {x+1,x+2,x+3,x+4}.issubset(rset) and (x not in rset): return "OESD"
    for x in range(1,11):
        if len({x,x+1,x+2,x+3,x+4} & rset)==4: return "Gutshot"
    return None

def overcards_count(hole,board):
    # Count number of hole cards higher than any board card
    if not board: return 0
    maxb=max(c.rank for c in board)
    return sum(1 for c in hole if c.rank>maxb)

def recommended_sizes(pot,to_call,stack):
    # Suggest bet and raise-to sizes relative to pot and stack
    base=pot if pot>0 else BIG_BLIND
    bet=int(max(BIG_BLIND,0.66*base)); bet=min(bet,stack)
    rto=int(to_call+0.66*(pot+to_call)); rto=min(max(rto,to_call+BIG_BLIND),stack)
    return max(bet,0), max(rto,to_call)

# ---- logs/analytics ----
HAND_LOGS=[]; TOURNEY_LOGS=[]

def log_action(hand_no, street, player, hole, board, pot_before, to_call, stack_hero, stack_opp, action, amount, equity=None, result=None):
    # Save one action into the hand log
    HAND_LOGS.append(dict(
        hand_no=hand_no,street=street,player=player,
        hole=fmt_cards(hole),board=fmt_cards(board),
        pot_before=pot_before,to_call=to_call,
        stack_hero=stack_hero,stack_opp=stack_opp,
        action=action,amount=amount,equity=equity,result=result,reward=None,
        hand_label=classify_hand(hole,board),
        flush_draw=has_flush_draw(hole,board)[0],
        straight_draw=straight_draw_type(hole,board),
        overcards=overcards_count(hole,board)
    ))

def fill_rewards_for_hand(hand_no, deltas): 
    # Backfill final rewards for each player's actions in a given hand
    for r in HAND_LOGS:
        if r["hand_no"]==hand_no and r["player"] in deltas: 
            r["reward"]=deltas[r["player"]]

def save_history_csv(path=SAVE_HISTORY_CSV):
    # Save full hand log to CSV (UTF-8 with BOM for Excel compatibility)
    if not HAND_LOGS: 
        print("No hand logs yet."); return
    try:
        if pd is not None: 
            pd.DataFrame(HAND_LOGS).to_csv(path,index=False,encoding="utf-8-sig")
        else:
            import csv
            keys=list(HAND_LOGS[0].keys())
            with open(path,"w",newline="",encoding="utf-8-sig") as f:
                w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
                for row in HAND_LOGS: w.writerow(row)
        print(f"Saved hand history to {os.path.abspath(path)}")
    except Exception as e:
        print(f"CSV save failed: {e}")

def analytics_report():
    # Compute aggregate player stats (VPIP, PFR, Aggression Factor, Reward)
    if not HAND_LOGS: return None
    players=set(r["player"] for r in HAND_LOGS if r["player"]!="RESULT")
    stats={}
    for p in players:
        rows=[r for r in HAND_LOGS if r["player"]==p]
        pre=[r for r in rows if r["street"]=="preflop"]
        post=[r for r in rows if r["street"] in ("flop","turn","river")]
        vpip=sum(1 for r in pre if r["action"] in ("call","bet","raise"))/max(len(pre),1)
        pfr=sum(1 for r in pre if r["action"] in ("bet","raise"))/max(len(pre),1)
        br=sum(1 for r in post if r["action"] in ("bet","raise"))
        calls=sum(1 for r in post if r["action"]=="call")
        af=(br/calls) if calls>0 else (float('inf') if br>0 else 0.0)
        reward=sum(r["reward"] for r in rows if r["reward"] is not None)
        stats[p]=dict(VPIP=vpip,PFR=pfr,AF=af,total_reward=reward)
    return stats

def plot_basic_charts():
    # Plot simple analytics if pandas + matplotlib available
    if pd is None or plt is None: 
        print("Install pandas+matplotlib for charts."); return
    df=pd.DataFrame(HAND_LOGS)
    if "reward" not in df.columns: 
        print("No rewards yet."); return
    # Cumulative winnings plot
    plt.figure()
    for player,g in df[df.player!="RESULT"].groupby("player"):
        s=g.groupby("hand_no")["reward"].sum().sort_index().cumsum()
        plt.plot(s.index,s.values,label=player)
    plt.title("Cumulative Winnings by Hand"); plt.xlabel("Hand #"); plt.ylabel("Cumulative Reward"); plt.legend(); plt.show()
    # Action frequency histogram
    plt.figure()
    df[df.player!="RESULT"]["action"].value_counts().plot(kind="bar")
    plt.title("Action Frequencies"); plt.xlabel("Action"); plt.ylabel("Count"); plt.show()

# ---- numpy utils/encoder ----
def _to_nd(x): return _np.array(x,dtype=float)  # Convert to NumPy float array

def _softmax(z): 
    # Numerically stable softmax over last dimension
    z=z-z.max(axis=1,keepdims=True); e=_np.exp(z); return e/e.sum(axis=1,keepdims=True)

def _one_hot(y,k): 
    # One-hot encode integer labels y into k classes
    Y=_np.zeros((len(y),k),dtype=float); Y[_np.arange(len(y)),y]=1.0; return Y

def _np_asarray(x,dtype=float): return _np.asarray(x,dtype=dtype)  # Thin wrapper for dtype control

def _stack(lst): return _np.vstack(lst)  # Stack list of arrays by rows

def _relu(x): return _np.maximum(0,x)  # ReLU activation

def _relu_grad(h): return (h>0).astype(float)  # ReLU derivative for backprop

def street_one_hot(n): 
    # One-hot encode street by #board cards: 0=pre, 3=flop, 4=turn, else river
    return [1,0,0,0] if n==0 else [0,1,0,0] if n==3 else [0,0,1,0] if n==4 else [0,0,0,1]

def encode_state(hole,board,pot,to_call,hero_stack,opp_stack,equity):
    # Build a compact numeric state vector for policy input
    s=street_one_hot(len(board))  # street phase
    potn=pot/max(BIG_BLIND*100,1); calln=to_call/max(pot+to_call,1)  # normalize pot & to-call
    hsn=hero_stack/max(STARTING_STACK*2,1); osn=(opp_stack or STARTING_STACK)/max(STARTING_STACK*2,1)  # stack ratios
    hranks=sorted([c.rank for c in hole],reverse=True); branks=sorted([c.rank for c in board],reverse=True)
    hr=[(r-2)/12 for r in hranks]+[0]*(2-len(hranks))  # hole ranks normalized (pad to 2)
    br=[((branks[i]-2)/12 if i<len(branks) else 0) for i in range(5)]  # board ranks normalized (pad to 5)
    suited=1.0 if len(hole)==2 and hole[0].suit==hole[1].suit else 0.0  # suited indicator
    connected=1.0 if len(hole)==2 and abs(hole[0].rank-hole[1].rank) in (1,2) else 0.0  # connector/gapper
    fdraw=1.0 if has_flush_draw(hole,board)[0] else 0.0  # flush draw flag
    sdt=straight_draw_type(hole,board); s_oesd=1.0 if sdt=="OESD" else 0.0; s_gut=1.0 if sdt=="Gutshot" else 0.0  # straight draws
    overc=overcards_count(hole,board)/2.0; eq=equity if equity is not None else 0.5  # overcards & equity
    # Concatenate all features and return as (1, D) for batch compatibility
    return _np_asarray(s+[potn,calln,hsn,osn]+hr+br+[suited,connected,fdraw,s_oesd,s_gut,overc,eq]).reshape(1,-1)

FEATURE_DIM = len(encode_state([Card(14,"S"),Card(13,"S")],[],0,0,STARTING_STACK,STARTING_STACK,0.5)[0])
# Dimensionality of state vector (computed with a dummy state)

# ---- action mapping ----
ACTIONS_FACING_BET=["fold","call","raise_s","raise_m","jam"]  # Discrete choices when facing a bet
ACTIONS_NO_BET=["check","bet_s","bet_m","bet_p","jam"]        # Discrete choices when no bet to you

def apply_action_mapping(idx,facing, pot,to_call,stack):
    # Translate discrete action index into concrete action + amount based on situation
    if facing:
        act=ACTIONS_FACING_BET[idx]
        if act=="fold": return "fold",0
        if act=="call": return "call", min(to_call,stack)
        if act=="raise_s": 
            tgt=int(to_call+0.5*(pot+to_call)); return "raise", min(max(BIG_BLIND,tgt-to_call),stack)
        if act=="raise_m": 
            tgt=int(to_call+0.75*(pot+to_call)); return "raise", min(max(BIG_BLIND,tgt-to_call),stack)
        if act=="jam": return "raise", stack
    else:
        act=ACTIONS_NO_BET[idx]
        if act=="check": return "check",0
        base=pot if pot>0 else BIG_BLIND
        if act=="bet_s": return "bet", min(max(BIG_BLIND,int(0.33*base)),stack)
        if act=="bet_m": return "bet", min(max(BIG_BLIND,int(0.66*base)),stack)
        if act=="bet_p": return "bet", min(max(BIG_BLIND,int(1.00*base)),stack)
        if act=="jam": return "bet", stack
    # Fallbacks (should rarely trigger)
    return ("check" if not facing else "call", min(to_call,stack))

def legal_action_mask(facing): 
    # Currently all 5 actions are allowed in both modes; mask is all zeros (no -inf)
    return _np.zeros((1,5), dtype=float)

# ---- players ----
class Player:
    def __init__(self,name,stack=STARTING_STACK):
        # Base player with name, stack, current hole cards, and in-hand flag
        self.name=name; self.stack=stack; self.hole=[]; self.in_hand=True
    def bet(self,amt): 
        # Deduct a bounded amount from stack and return the actual paid amount
        amt=max(0,min(int(amt),self.stack)); self.stack-=amt; return amt

class HeuristicAI(Player):
    def decide(self,board,pot,to_call,min_raise,street):
        # Simple Monte Carlo equity vs. pot odds heuristic for betting decisions
        trials=MC_TRIALS[street]; equity=monte_carlo_equity(self.hole,board,1,trials)
        pot_odds=to_call/(pot+to_call) if to_call>0 else 0.0; margin=0.10 if street=="preflop" else 0.07
        if to_call==0:
            # Choose a value/semi-bluff size relative to pot & equity edge
            base=max(BIG_BLIND,int((pot or BIG_BLIND)*max(0.25,equity-0.45)))
            size=max(min_raise,base); size=min(size,self.stack)
            return ("check",0,equity) if size<=0 else ("bet",size,equity)
        if equity>pot_odds+margin:
            # Raise when equity edge exceeds odds by a margin
            raise_amt=max(min_raise,int((pot+to_call)*0.6)); raise_amt=min(raise_amt,self.stack)
            return ("call",min(to_call,self.stack),equity) if raise_amt<=0 else ("raise",raise_amt,equity)
        if equity>=pot_odds: return ("call",min(to_call,self.stack),equity)  # Break-even call
        if to_call<=max(BIG_BLIND,int(0.03*self.stack)): return ("call",min(to_call,self.stack),equity)  # Cheap call
        return ("fold",0,equity)  # Otherwise fold

# ---- numpy policy + RL ----
class MLPPolicy:
    def __init__(self,input_dim,hidden=64,seed=None):
        # Lightweight NumPy MLP policy: input -> hidden -> logits(5 actions)
        rnd=random.Random(seed)
        def xavier(i,o): 
            # Xavier/Glorot uniform init
            s=math.sqrt(6/(i+o)); return [[rnd.uniform(-s,s) for _ in range(o)] for _ in range(i)]
        self.W1=_to_nd(xavier(input_dim,hidden)); self.b1=_np.zeros((hidden,))
        self.W2=_to_nd(xavier(hidden,5)); self.b2=_np.zeros((5,))
    def forward(self,x): 
        # Forward pass returns logits and hidden activation
        z1=x@self.W1+self.b1; h=_relu(z1); z2=h@self.W2+self.b2; return z2,h
    def probs(self,x,mask=None): 
        # Convert logits to action probabilities (optionally add mask)
        logits,h=self.forward(x); logits=logits+(mask if mask is not None else 0); return _softmax(logits)
    def step_supervised(self,X,y,lr=LEARNING_RATE):
        # One supervised gradient step (cross-entropy) on batch (X,y)
        logits,h=self.forward(X); P=_softmax(logits); N=X.shape[0]; Y=_one_hot(y,5)
        d=(P-Y)/max(N,1); dW2=h.T@d; db2=d.sum(axis=0); dh=d@self.W2.T; dz=dh*_relu_grad(h); dW1=X.T@dz; db1=dz.sum(axis=0)
        self.W2-=lr*dW2; self.b2-=lr*db2; self.W1-=lr*dW1; self.b1-=lr*db1
    def step_reinforce(self,X,A,G,mask_list=None,lr=LEARNING_RATE):
        # REINFORCE: policy gradient with advantages G and optional action mask
        logits,h=self.forward(X); logits=logits+(mask_list if mask_list is not None else 0); P=_softmax(logits); N=X.shape[0]
        d=P; d[_np.arange(N),A]-=1.0; d=(d*G.reshape(-1,1))/max(N,1)  # scale by advantages
        dW2=h.T@d; db2=d.sum(axis=0); dh=d@self.W2.T; dz=dh*_relu_grad(h); dW1=X.T@dz; db1=dz.sum(axis=0)
        self.W2-=lr*dW2; self.b2-=lr*db2; self.W1-=lr*dW1; self.b1-=lr*db1

class RLAgent(Player):
    def __init__(self,name,policy:MLPPolicy,stack=STARTING_STACK):
        # On-policy episodic learner using the NumPy MLP
        super().__init__(name,stack); self.policy=policy
        self.episode_states=[]; self.episode_actions=[]; self.episode_masks=[]; self.baseline=0.0
    def decide(self,board,pot,to_call,min_raise,street):
        # Sample an action from current policy given encoded state
        eq=monte_carlo_equity(self.hole,board,1,MC_TRIALS[street])
        x=encode_state(self.hole,board,pot,to_call,self.stack,0,eq); facing=(to_call>0); mask=legal_action_mask(facing)
        probs=self.policy.probs(x,mask=mask); a=int(_np.random.choice(5,p=probs[0]))
        action,amt=apply_action_mapping(a,facing,pot,to_call,self.stack)
        # Store transition for learning later
        self.episode_states.append(x[0]); self.episode_actions.append(a); self.episode_masks.append(mask[0])
        return (action,amt,eq)
    def learn_from_reward(self,reward):
        # Apply a single REINFORCE update using final hand reward with a moving baseline
        if not self.episode_states: return
        X=_np_asarray(self.episode_states,dtype=float); A=_np_asarray(self.episode_actions,dtype=int); M=_np_asarray(self.episode_masks,dtype=float)
        self.baseline=RL_BASELINE_MOMENTUM*self.baseline+(1-RL_BASELINE_MOMENTUM)*reward
        adv=reward-self.baseline; G=_np_asarray([adv]*len(A),dtype=float)
        self.policy.step_reinforce(X,A,G,mask_list=M,lr=LEARNING_RATE)
        # Clear episode buffer
        self.episode_states.clear(); self.episode_actions.clear(); self.episode_masks.clear()

# ---- torch actor-critic ----
if TORCH_AVAILABLE:
    class TorchPolicy(nn.Module):
        def __init__(self,input_dim,hidden=TORCH_HIDDEN):
            # Two-layer MLP feature extractor -> policy head (pi) and value head (v)
            super().__init__(); self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Linear(hidden,64),nn.ReLU())
            self.pi=nn.Linear(64,5); self.v=nn.Linear(64,1)  # 5 discrete actions; scalar state value
        def forward(self,x,mask=None):
            # Forward pass: compute logits for actions and the state value
            h=self.net(x); logits=self.pi(h)
            if mask is not None: logits=logits+mask  # optional action mask (additive)
            return logits, self.v(h).squeeze(-1)  # logits: [B,5], value: [B]

    class TorchA2CAgent(Player):
        def __init__(self,name,policy:TorchPolicy,stack=STARTING_STACK,lr=TORCH_LR,entropy_beta=TORCH_ENTROPY_BETA):
            # A2C agent with entropy regularization and tiny replay for stability
            super().__init__(name,stack); self.policy=policy; self.opt=optim.Adam(policy.parameters(),lr=lr)
            self.entropy_beta=entropy_beta; self.traj=[]; self.replay=deque(maxlen=REPLAY_CAPACITY); self.device=next(policy.parameters()).device
        def decide(self,board,pot,to_call,min_raise,street):
            # Build state features and sample an action from the policy distribution
            eq=monte_carlo_equity(self.hole,board,1,MC_TRIALS[street])
            s=encode_state(self.hole,board,pot,to_call,self.stack,0,eq); facing=(to_call>0); mask=legal_action_mask(facing)
            x=torch.tensor(s,dtype=torch.float32,device=self.device); m=torch.tensor(mask,dtype=torch.float32,device=self.device)
            logits,val=self.policy(x,mask=m); probs=torch.softmax(logits,dim=-1); dist=torch.distributions.Categorical(probs=probs)
            a=int(dist.sample()[0].item()); logp=dist.log_prob(torch.tensor([a],device=self.device)).squeeze(0); ent=dist.entropy().mean()
            action,amt=apply_action_mapping(a,facing,pot,to_call,self.stack)  # map discrete action -> (verb, amount)
            # Save trajectory step for learning
            self.traj.append((logp,val.squeeze(0),ent,s[0],mask[0],a)); return (action,amt,eq)
        def learn_from_reward(self,reward):
            # Single-step return A2C update using episode reward as return
            if not self.traj: return
            returns=torch.tensor([reward]*len(self.traj),dtype=torch.float32,device=self.device)
            logps=torch.stack([t[0] for t in self.traj]); values=torch.stack([t[1] for t in self.traj]); ents=torch.stack([t[2] for t in self.traj])
            adv=returns-values.detach(); loss=-(logps*adv).mean() - self.entropy_beta*ents.mean() + 0.5*(returns-values).pow(2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            # Add to tiny replay buffer for occasional extra updates
            for (_,_,_,s,mk,a) in self.traj: self.replay.append((s,mk,a,reward))
            self._replay_step(); self.traj.clear()
        def _replay_step(self):
            # Optional small replay batch to smooth learning noise
            if len(self.replay)<REPLAY_BATCH: return
            batch=random.sample(self.replay,REPLAY_BATCH)
            s=torch.tensor([b[0] for b in batch],dtype=torch.float32,device=self.device)
            m=torch.tensor([b[1] for b in batch],dtype=torch.float32,device=self.device)
            a=torch.tensor([b[2] for b in batch],dtype=torch.long,device=self.device)
            R=torch.tensor([b[3] for b in batch],dtype=torch.float32,device=self.device)
            logits,v=self.policy(s,mask=m); probs=torch.softmax(logits,dim=-1); dist=torch.distributions.Categorical(probs=probs)
            logp=dist.log_prob(a); ent=dist.entropy().mean(); adv=(R-v.detach())
            loss=-(logp*adv).mean()+0.5*(R-v).pow(2).mean()-TORCH_ENTROPY_BETA*ent
            self.opt.zero_grad(); loss.backward(); self.opt.step()
else:
    # Stubs keep imports safe when torch is not installed
    class TorchPolicy: 
        def __init__(self,*a,**k): raise RuntimeError("PyTorch not available. Install torch to use Torch modes.")
    class TorchA2CAgent(Player):
        def __init__(self,*a,**k): raise RuntimeError("PyTorch not available. Install torch to use Torch modes.")

# ---- input / rules ----
def input_yes_no(prompt):
    # Read a yes/no response from stdin; re-prompt on invalid input
    while True:
        s=input(prompt+" (Y/N): ").strip().lower()
        if s in ("y","yes"): return True
        if s in ("n","no"): return False
        print("Please enter Y or N.")

def input_int(prompt,min_val,max_val):
    # Read an integer in [min_val, max_val]; re-prompt on invalid input
    while True:
        s=input(f"{prompt} [{min_val}-{max_val}]: ").strip()
        if s.isdigit():
            v=int(s)
            if min_val<=v<=max_val: return v
        print("Please enter a valid number in range.")

def print_rules():
    """
    Comprehensive Texas Hold'em rules and strategy guide for new players.
    Covers game mechanics, hand rankings, betting actions, and basic strategy.
    Designed to be educational while remaining practical for actual gameplay.
    """
    # Header with visual separator for better readability
    print("\n" + "="*60)
    print("TEXAS HOLD'EM - COMPLETE RULES & STRATEGY GUIDE")
    print("="*60)
    
    # Main objective section - explains core win condition and hand formation
    print("\n🎯 GAME OBJECTIVE")
    print("-" * 20)
    print("Make the best possible 5-card poker hand using any combination of:")
    print("• Your 2 private hole cards (only you can see these)")
    print("• The 5 community cards (shared by all players)")
    print("• You must use exactly 5 cards total for your final hand")
    
    # Hand rankings from strongest to weakest with concrete examples
    # Critical for new players to understand what beats what
    print("\nHAND RANKINGS (Highest to Lowest)")
    print("-" * 40)
    print("1. STRAIGHT FLUSH: Five cards in sequence, all same suit (A♠ K♠ Q♠ J♠ 10♠)")
    print("2. FOUR OF A KIND: Four cards of same rank (K♥ K♠ K♦ K♣ 9♥)")
    print("3. FULL HOUSE: Three of a kind + a pair (Q♥ Q♠ Q♦ 7♣ 7♥)")
    print("4. FLUSH: Five cards of same suit, not in sequence (A♠ J♠ 9♠ 5♠ 3♠)")
    print("5. STRAIGHT: Five cards in sequence, mixed suits (8♥ 7♠ 6♦ 5♣ 4♥)")
    print("6. THREE OF A KIND: Three cards of same rank (10♥ 10♠ 10♦ A♣ 5♥)")
    print("7. TWO PAIR: Two pairs of different ranks (A♥ A♠ 8♦ 8♣ K♥)")
    print("8. ONE PAIR: Two cards of same rank (J♥ J♠ A♦ 9♣ 5♥)")
    print("9. HIGH CARD: No pairs, highest card wins (A♥ K♠ Q♦ 8♣ 4♥)")
    
    # Game flow section - explains the 4 betting rounds chronologically
    # Each round has different strategic implications and betting patterns
    print("\n🎮 GAME FLOW & BETTING ROUNDS")
    print("-" * 35)
    print("Each hand has up to 4 betting rounds:")
    
    # Preflop: Most important round for hand selection
    print("\n1️⃣  PREFLOP:")
    print("   • Each player gets 2 hole cards face down")
    print("   • Small Blind posts $5, Big Blind posts $10")
    print("   • Action starts with Small Blind (you or AI)")
    print("   • Players can: Fold, Call $10, or Raise")
    
    # Flop: 60% of final hand strength is revealed
    print("\n2️⃣  FLOP:")
    print("   • 3 community cards dealt face up")
    print("   • First betting round after seeing the flop")
    print("   • Action starts with Big Blind position")
    print("   • Players can: Check, Bet, Call, Raise, or Fold")
    
    # Turn: Single card can drastically change hand strength
    print("\n3️⃣  TURN:")
    print("   • 1 more community card dealt (4 total)")
    print("   • Another betting round with same actions available")
    print("   • Bet sizes typically larger than flop")
    
    # River: Final betting round, no more cards coming
    print("\n4️⃣  RIVER:")
    print("   • Final community card dealt (5 total)")
    print("   • Last betting round")
    print("   • If multiple players remain, showdown occurs")
    
    # Betting actions explained - fundamental mechanics every player must understand
    # Understanding these actions is crucial for making strategic decisions
    print("\n💰 BETTING ACTIONS EXPLAINED")
    print("-" * 30)
    print("FOLD: Give up your cards and forfeit any money already bet")
    print("CHECK: Pass the action without betting (only when no bet to call)")
    print("CALL: Match the current bet amount")
    print("BET: Put money in the pot when no one has bet yet")
    print("RAISE: Increase the current bet (must be at least double)")
    print("ALL-IN: Bet all remaining chips")
    
    # Heads-up specific rules - different from full-table poker
    # Important distinctions that affect strategy and position play
    print("\n🎯 HEADS-UP SPECIFIC RULES")
    print("-" * 28)
    print("• Only 2 players (you vs AI)")
    print("• Button player posts Small Blind ($5) and acts first preflop")
    print("• Big Blind player acts first on all other streets")
    print("• Button rotates each hand")
    print("• No-Limit: You can bet any amount up to your stack")
    
    # Starting conditions - sets expectations for session length and stakes
    # Uses actual game constants for accuracy
    print("\n📊 STARTING CONDITIONS")
    print("-" * 22)
    print(f"• Both players start with ${STARTING_STACK} in chips")  # Dynamic from config
    print(f"• Small Blind: ${SMALL_BLIND}")  # Dynamic from config
    print(f"• Big Blind: ${BIG_BLIND}")  # Dynamic from config
    print("• Chips carry over between hands")
    print("• Game ends when someone runs out of chips or session limit reached")
    
    # Basic strategy section - teaches fundamental poker concepts
    # These tips help beginners avoid common mistakes and play more effectively
    print("\n🧠 BASIC STRATEGY TIPS")
    print("-" * 23)
    print("POSITION: Button (Small Blind) has advantage - acts last on flop/turn/river")
    print("STARTING HANDS: Play tight preflop - premium pairs, big aces, suited connectors")
    print("POT ODDS: Call when your winning chances exceed the pot odds")
    print("AGGRESSION: Betting and raising is usually better than calling")
    print("BLUFFING: Don't bluff too much - pick your spots carefully")
    print("BANKROLL: Don't risk more than 5% of your stack on speculative hands")
    
    # Practical scenarios section - concrete examples help players apply concepts
    # Shows real decision-making situations they'll encounter in gameplay
    print("\n🎲 EXAMPLE SCENARIOS")
    print("-" * 20)
    # Strong preflop hand in position - straightforward value play
    print("SCENARIO 1 - Preflop:")
    print("You have A♥ K♠ on the button. AI posts BB ($10).")
    print("→ RAISE to $25-30 (strong hand, good position)")
    
    # Value betting with top pair - extract value from weaker hands
    print("\nSCENARIO 2 - Flop:")
    print("Board: A♠ 8♥ 3♦. You have A♥ K♠ (top pair, good kicker).")
    print("AI checks. → BET $15-20 (value bet with strong hand)")
    
    # Fold weak hand facing aggression - avoid costly mistakes
    print("\nSCENARIO 3 - Turn:")
    print("Board: A♠ 8♥ 3♦ J♣. You have 5♥ 4♥ (no pair, no draws).")
    print("AI bets $30. → FOLD (weak hand, facing aggression)")
    
    # Important notes section - covers technical and gameplay details
    # Addresses common questions and clarifies game mechanics
    print("\n⚠️  IMPORTANT NOTES")
    print("-" * 18)
    print("• Cards are shuffled randomly each hand")  # Ensures fairness
    print("• AI difficulty varies by mode (Heuristic/Learning/Advanced)")  # Sets expectations
    print("• Hand history is logged for analysis")  # Data collection feature
    print("• Enable 'Advice Mode' for equity calculations and tips")  # Learning aid
    print("• This is No-Limit Hold'em - bet sizing is crucial")  # Strategic emphasis
    
    # Closing footer with encouragement and responsible gaming reminder
    # Maintains positive tone while promoting healthy play habits
    print("\n" + "="*60)
    print("Good luck at the tables! Remember: poker is a game of skill,")
    print("patience, and calculated risks. Play responsibly.")
    print("="*60)
# ---- coaching (post-loss) ----
def coach_from_stats(stats):
    # Generate quick coaching tips from summary stats (VPIP, PFR, AF)
    tips=[]
    vpip=stats["VPIP"]; pfr=stats["PFR"]; af=stats["AF"]
    if vpip<0.15: tips.append("You're too tight preflop. Open more playable hands (suited connectors, broadways).")
    if vpip>0.45: tips.append("You're too loose preflop. Tighten up offsuit junk and weak gappers.")
    if pfr<0.5*max(vpip,1e-9): tips.append("Too many limps/calls. Raise more of the hands you enter.")
    if af<1.0: tips.append("Postflop passive. Add more value bets and semi-bluffs instead of calling.")
    if af>4.0: tips.append("Overly aggressive. Slow down multi-street bluffs and out-of-position barrels.")
    return tips or ["Review your biggest losing hands: check sizing discipline and position awareness."]

# ---- game/session ----
class TexasHoldemGame:
    def __init__(self,user_name,stacks=(STARTING_STACK,STARTING_STACK),ai_mode="heuristic",torch_device="cpu",torch_policy=None):
        # Instantiate opponents according to selected mode
        if ai_mode=="rl_numpy":
            self.policy_np=MLPPolicy(FEATURE_DIM,hidden=64,seed=42); ai=RLAgent(AI_NAME,self.policy_np,stack=stacks[1])
        elif ai_mode=="rl_torch" and TORCH_AVAILABLE:
            self.policy_torch=torch_policy or TorchPolicy(FEATURE_DIM,hidden=TORCH_HIDDEN)
            if TORCH_AVAILABLE and hasattr(self.policy_torch,"to"): self.policy_torch=self.policy_torch.to(torch_device)
            ai=TorchA2CAgent(AI_NAME,self.policy_torch,stack=stacks[1])
        else:
            # Fallback to heuristic if mode unavailable
            self.policy_np=None; self.policy_torch=None; ai=HeuristicAI(AI_NAME,stack=stacks[1]); ai_mode="heuristic"
        # Player 0 is human; player 1 is AI
        self.players=[Player(user_name,stacks[0]), ai]; self.ai_mode=ai_mode; self.button=0; self.hand_no=0; self._snap={}
    def play_hand(self,advice=False):
        # Run a single hand from blinds through river/showdown
        self.hand_no+=1; hand_no=self.hand_no; self._snap[hand_no]=(self.players[0].stack,self.players[1].stack)
        deck=Deck(); pot=0; board=[]
        for p in self.players: p.in_hand=True; p.hole=[]
        sb=self.button; bb=self.button^1  # heads-up button indexes
        pot+=self.players[sb].bet(SMALL_BLIND); pot+=self.players[bb].bet(BIG_BLIND)  # post blinds
        to_call={sb:BIG_BLIND-SMALL_BLIND, bb:0}  # preflop open action: SB owes the difference
        for p in self.players: p.hole=deck.deal(2)  # deal 2 cards to each player

        # PATCHED: ask_action now takes `need` from the live `current` dict
        def ask_action(i, min_raise, street, need):
            p=self.players[i]
            if not p.in_hand: return ("fold",0)  # If player already folded, skip their turn

            print("\n----------------------------------------")
            print(f"Street: {street.upper()}  |  Pot: ${pot}")
            print(f"Board: {fmt_cards(board)}")
            print(f"{self.players[0].name} Stack: ${self.players[0].stack}   {self.players[1].name} Stack: ${self.players[1].stack}")

            # --- AI (opponent) branch ---
            if i==1:
                # Get AI's decision given the *current* amount to call (`need`)
                action,amt,eq=self.players[1].decide(board,pot,need,min_raise,street)
                # Log the AI action with equity snapshot for analytics
                log_action(hand_no,street,self.players[1].name,self.players[1].hole,board,pot,need,self.players[1].stack,self.players[0].stack,action,amt,equity=eq)
                # Print a human-readable narration of the AI's action
                if action in ("fold","check","call"):
                    msg={"fold":"folds.","check":"checks.","call":f"calls ${amt}."}[action]; print(f"{self.players[1].name} {msg}")
                else:
                    # For bet/raise, show the final total amount (raise is over `need`)
                    verb="bets" if action=="bet" else "raises"; to_amt=(need+amt) if need>0 else amt
                    print(f"{self.players[1].name} {verb} to ${to_amt}.")
                return (action,amt)

            # --- Human (hero) branch ---
            print(f"Your cards: {fmt_cards(p.hole)}")
            eq=None
            if advice:
                # Compute quick Monte Carlo equity and basic pot odds guidance
                k="preflop" if len(board)==0 else "flop" if len(board)==3 else "turn" if len(board)==4 else "river"
                eq=monte_carlo_equity(p.hole,board,1,MC_TRIALS[k]); po=(need/(pot+need)) if need>0 else 0.0
                print("\n[AI-Powered Advice]")
                print(f"Hand: {classify_hand(p.hole,board)} | Equity≈{eq:.1%} vs Pot Odds {po:.1%} (need ≥ {po:.1%})")
                # Show draws and typical out counts for quick intuition
                fd,_=has_flush_draw(p.hole,board); sdt=straight_draw_type(p.hole,board); oc=overcards_count(p.hole,board)
                draws=[]
                if fd: draws.append("flush draw (~9 outs)")
                if sdt=="OESD": draws.append("OESD (~8 outs)")
                elif sdt=="Gutshot": draws.append("gutshot (~4 outs)")
                if oc>0: draws.append(f"{oc} overcard(s)")
                if draws: print("Draws: "+", ".join(draws))
                # Recommend a size based on pot/need/stack heuristics
                bsz,rto=recommended_sizes(pot,need,p.stack)
                print(f"Suggested {'bet' if need==0 else 'raise-to'}: ${bsz if need==0 else rto}")

            # No bet facing us: we can check, bet, or fold (fold is odd but allowed for symmetry)
            if need==0:
                while True:
                    c=input("Your action? (check/bet/fold): ").strip().lower()
                    if c in ("fold","f"): 
                        # Log and fold immediately
                        log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"fold",0,equity=eq); 
                        return ("fold",0)
                    if c in ("check","x"): 
                        # Check keeps pot size the same
                        log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"check",0,equity=eq); 
                        return ("check",0)
                    if c in ("bet","b"):
                        # Validate bet size within [min_raise, stack]
                        min_b=max(BIG_BLIND,min_raise); max_b=p.stack
                        if max_b<=0: 
                            # If we can't bet, treat as a check
                            log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"check",0,equity=eq); 
                            return ("check",0)
                        amt=input_int("Enter bet amount",min_b,max_b)
                        log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"bet",amt,equity=eq); 
                        return ("bet",amt)
                    # Loop until a valid action is provided
                    print("Please choose: check, bet, or fold.")

            # We are facing a bet: fold, call (to `need`), or raise (add on top of `need`)
            else:
                while True:
                    c=input(f"Your action? (fold/call/raise) [to_call=${need}]: ").strip().lower()
                    if c in ("fold","f"): 
                        # Fold and end our participation in the hand
                        log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"fold",0,equity=eq); 
                        return ("fold",0)
                    if c in ("call","c"): 
                        # Call exactly `need`
                        log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"call",need,equity=eq); 
                        return ("call",need)
                    if c in ("raise","r"):
                        # Ask for a raise amount *over* the call (i.e., additional chips beyond `need`)
                        max_r=p.stack
                        if max_r<=0:
                            # If we can't afford a raise, auto-call to keep game flowing
                            print("You cannot raise; calling instead.")
                            log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"call",need,equity=eq); 
                            return ("call",need)
                        min_r=max(min_raise,BIG_BLIND); 
                        amt=input_int("Enter raise amount over call",min_r,max_r)
                        log_action(hand_no,street,p.name,p.hole,board,pot,need,p.stack,self.players[1].stack,"raise",amt,equity=eq); 
                        return ("raise",amt)
                    # Loop until a valid action is given
                    print("Please choose: fold, call, or raise.")

        def betting_round(start_i,min_open,street):
            nonlocal pot
            # Track current "to call" per player; `acted` tells if both have taken an action at this level
            min_raise=max(min_open,BIG_BLIND); current=dict(to_call); acted=set(); i=start_i; last_raise=min_raise
            def settled(): return current[0]==0 and current[1]==0 and len(acted)==2  # betting round ends when both called/checked
            while True:
                if not self.players[i].in_hand:
                    # Skip players who already folded
                    acted.add(i); i^=1
                    if len(acted)==2: break
                    continue
                need=current[i]
                # PATCHED: pass the live `need` into ask_action
                act,amt=ask_action(i,last_raise,street,need)
                if act=="fold": 
                    # Player folds; end the round early (pot awarded later)
                    self.players[i].in_hand=False; 
                    break
                elif act=="check":
                    # If they "check" while there is `need`, auto-correct by paying it as a call (keeps game robust)
                    if need!=0: 
                        pay=self.players[i].bet(need); pot+=pay; current[i]=0
                    acted.add(i)
                elif act=="call":
                    # Pay exactly `need`, then mark this player as acted
                    pay=self.players[i].bet(need); pot+=pay; current[i]=0; acted.add(i)
                else:
                    # Bet/Raise: pay `need` + added amount; opponent now faces that total
                    pay=need+amt; pay=self.players[i].bet(pay); pot+=pay; current[i]=0
                    j=i^1; current[j]=need+amt; acted={i}; last_raise=max(BIG_BLIND,amt)
                # Switch to the other player
                i^=1
                if settled(): break
            return pot,current

        # Preflop betting
        pot,to_call=betting_round(sb,BIG_BLIND,"preflop")
        if sum(p.in_hand for p in self.players)<2:
            # Hand ended by a fold preflop: award pot and record results
            winner=0 if self.players[0].in_hand else 1; self.players[winner].stack+=pot
            snap=self._snap.get(hand_no,(self.players[0].stack,self.players[1].stack))
            d0=self.players[0].stack-snap[0]; d1=self.players[1].stack-snap[1]
            fill_rewards_for_hand(hand_no,{self.players[0].name:d0,self.players[1].name:d1})
            HAND_LOGS.append(dict(hand_no=hand_no,street="result",player="RESULT",hole="",board="—",pot_before=pot,to_call=0,
                                stack_hero=self.players[0].stack,stack_opp=self.players[1].stack,action=("win:"+self.players[winner].name),
                                amount=0,equity=None,result="win",reward=None,hand_label="",flush_draw=False,straight_draw=None,overcards=0))
            # Learning agent receives terminal reward signal
            if isinstance(self.players[1],(RLAgent,)) or (TORCH_AVAILABLE and isinstance(self.players[1],TorchA2CAgent)): self.players[1].learn_from_reward(d1)
            return dict(board=[],pot=pot,winner=winner)

        # Postflop streets (deal community cards and run a betting round each time)
        for n,name,start in [(3,"flop",bb),(1,"turn",bb),(1,"river",bb)]:
            board+=deck.deal(n); to_call={0:0,1:0}; pot,to_call=betting_round(start,BIG_BLIND,name)
            if sum(p.in_hand for p in self.players)<2:
                # Someone folded during this street; award pot and stop dealing further cards
                winner=0 if self.players[0].in_hand else 1; self.players[winner].stack+=pot; break
        # If both players still in at the end, showdown: evaluate best hands and split/take pot
        if sum(p.in_hand for p in self.players)>=2:
            s0=HandEvaluator.evaluate(self.players[0].hole,board); s1=HandEvaluator.evaluate(self.players[1].hole,board)
            if s0>s1: winner=0; self.players[0].stack+=pot
            elif s1>s0: winner=1; self.players[1].stack+=pot
            else: winner=None; self.players[0].stack+=pot//2; self.players[1].stack+=pot-(pot//2)
        # Log final result line for analytics/CSV
        HAND_LOGS.append(dict(hand_no=hand_no,street="result",player="RESULT",hole="",board=fmt_cards(board),pot_before=pot,to_call=0,
                            stack_hero=self.players[0].stack,stack_opp=self.players[1].stack,action=("win:"+self.players[winner].name if winner is not None else "tie"),
                            amount=0,equity=None,result=("win" if winner is not None else "tie"),reward=None,hand_label="",flush_draw=False,straight_draw=None,overcards=0))
        # Backfill per-action rewards from the final chip delta snapshot
        snap=self._snap.get(hand_no,(self.players[0].stack,self.players[1].stack))
        d0=self.players[0].stack-snap[0]; d1=self.players[1].stack-snap[1]
        fill_rewards_for_hand(hand_no,{self.players[0].name:d0,self.players[1].name:d1})
        # Push reward to learning agents immediately after the hand
        if isinstance(self.players[1],(RLAgent,)) or (TORCH_AVAILABLE and isinstance(self.players[1],TorchA2CAgent)): self.players[1].learn_from_reward(d1)
        return dict(board=board,pot=pot,winner=winner)
    def rotate_button(self): self.button^=1

class Session:
    def __init__(self,user_name="You",hands=10,stacks=(STARTING_STACK,STARTING_STACK),ai_mode="heuristic",torch_device="cpu",torch_policy=None):
        # Create a game instance and store session length
        self.game=TexasHoldemGame(user_name,stacks=stacks,ai_mode=ai_mode,torch_device=torch_device,torch_policy=torch_policy); self.hands=hands
    def play(self,advice=False):
        # Run a multi-hand session loop
        for h in range(1,self.hands+1):
            # Hand header and who has the dealer button
            print("\n\n========================================"); print(f"Hand {h}  (Dealer/Button: {self.game.players[self.game.button].name})")
            # Play a single hand; returns dict(board, pot, winner)
            res=self.game.play_hand(advice=advice)
            p0,p1=self.game.players
            # Show result summary
            print("\n--- Hand Result ---")
            print(f"Board: {fmt_cards(res['board'])}")
            print(f"{p0.name} hole: {fmt_cards(p0.hole)}   | Stack: ${p0.stack}")
            print(f"{p1.name} hole: {fmt_cards(p1.hole)}   | Stack: ${p1.stack}")
            if res['winner'] is None: print(f"Result: TIE. Pot ${res['pot']} split.")
            else: print(f"Result: {self.game.players[res['winner']].name} wins ${res['pot']}.")
            # End session early if someone is busted
            if p0.stack<=0 or p1.stack<=0: print("\nGame over: someone is out of chips."); break
            # Move the button for next hand
            self.game.rotate_button()
        # Session footer + final stacks
        print("\n========================================"); print("Session complete!")
        print(f"Final stacks — {self.game.players[0].name}: ${self.game.players[0].stack} | {self.game.players[1].name}: ${self.game.players[1].stack}")
        # Aggregate basic player statistics
        stats=analytics_report()
        if stats:
            print("\n--- Player Stats ---")
            for p,st in stats.items():
                af="∞" if math.isinf(st["AF"]) else f"{st['AF']:.2f}"
                print(f"{p}: VPIP {st['VPIP']*100:.1f}%, PFR {st['PFR']*100:.1f}%, AF {af}, Total Reward {st['total_reward']:+.0f}")
            # If user lost, print tailored coaching tips
            you=self.game.players[0].name; opp=self.game.players[1].name
            if stats[you]["total_reward"]<stats[opp]["total_reward"]:
                print("\n--- Coaching Tips (you lost this session) ---")
                for t in coach_from_stats(stats[you]): print("• "+t)
        # Optional exports/plots
        if input_yes_no("\nSave hand history to CSV?"): save_history_csv(SAVE_HISTORY_CSV)
        if input_yes_no("Show basic charts (requires pandas + matplotlib)?"): plot_basic_charts()

# ---- dataset + training/eval ----
def generate_heuristic_dataset(n_hands=300):
    # Collect (state, action, mask) triples by letting two heuristics play
    X_list,y_list,M_list=[],[],[]
    tmp=TexasHoldemGame("Learner", ai_mode="heuristic")
    tmp.players=[HeuristicAI("H1",STARTING_STACK), HeuristicAI("H2",STARTING_STACK)]
    for _ in range(n_hands):
        deck=Deck(); pot=0; board=[]
        # Reset players for the synthetic hand
        for p in tmp.players: p.in_hand=True; p.hole=[]
        sb=tmp.button; bb=tmp.button^1
        # Post blinds
        pot+=tmp.players[sb].bet(SMALL_BLIND); pot+=tmp.players[bb].bet(BIG_BLIND)
        to_call={sb:BIG_BLIND-SMALL_BLIND, bb:0}
        # Deal hole cards
        for p in tmp.players: p.hole=deck.deal(2)
        def snapshot(i,street):
            # Encode current decision state for player i on given street
            p=tmp.players[i]; opp=tmp.players[i^1]; need=to_call[i]
            eq=monte_carlo_equity(p.hole,board,1,MC_TRIALS[street])
            x=encode_state(p.hole,board,pot,need,p.stack,opp.stack,eq); facing=(need>0); mask=legal_action_mask(facing)
            # Get the heuristic action to label this state
            action,amt,_=tmp.players[i].decide(board,pot,need,BIG_BLIND,street)
            if facing:
                # Map facing-bet actions to 5-class label
                mapping={"fold":0,"call":1,"raise":2}; yi=mapping.get(action,1)
                # Heuristic split between small/mid raise, and jam bucket
                if action=="raise": yi=3 if amt>=max(BIG_BLIND,0.6*(pot+need)) else 2
                if action in ("raise","bet") and amt>=p.stack: yi=4
            else:
                # Map no-bet actions to 5-class label
                mapping={"check":0,"bet":2}; yi=mapping.get(action,0)
                if action=="bet":
                    base=pot if pot>0 else BIG_BLIND; frac=amt/max(base,1)
                    # Bucketize bet sizes into small/medium/pot/jam categories
                    yi=1 if frac<=0.4 else 2 if frac<=0.8 else 3 if frac<1.2 else 4
            return x[0],yi,mask[0]
        def quick_round(start_i,street):
            # Advance a betting street while recording (state, label, mask)
            nonlocal pot
            current=dict(to_call); acted=set(); i=start_i
            while True:
                if not tmp.players[i].in_hand:
                    acted.add(i); i^=1
                    if len(acted)==2: break
                    continue
                # Record a snapshot for supervised learning
                x,yi,mk=snapshot(i,street); X_list.append(x); y_list.append(yi); M_list.append(mk)
                need=current[i]; action,amt,_=tmp.players[i].decide(board,pot,need,BIG_BLIND,street)
                # Apply action to toy pot/turn structure
                if action=="fold": tmp.players[i].in_hand=False; break
                elif action=="check":
                    if need!=0: pay=tmp.players[i].bet(need); pot+=pay; current[i]=0
                    acted.add(i)
                elif action=="call":
                    pay=tmp.players[i].bet(need); pot+=pay; current[i]=0; acted.add(i)
                else:
                    pay=need+amt; pay=tmp.players[i].bet(pay); pot+=pay; current[i]=0; j=i^1; current[j]=need+amt; acted={i}
                i^=1
                # Stop when both are settled for the street
                if current[0]==0 and current[1]==0 and len(acted)==2: break
        # Preflop collection
        quick_round(sb,"preflop")
        if sum(p.in_hand for p in tmp.players)<2: tmp.button^=1; continue
        # Flop/Turn/River collection
        for n,name,start in [(3,"flop",bb),(1,"turn",bb),(1,"river",bb)]:
            board+=deck.deal(n); to_call={0:0,1:0}; quick_round(start,name)
            if sum(p.in_hand for p in tmp.players)<2: break
        # Swap button each synthetic hand
        tmp.button^=1
    # Stack into arrays for training
    X=_stack(X_list); y=_np.asarray(y_list,dtype=int); M=_stack(M_list); return X,y,M

def split_train_val(X,y,M,val_split=VAL_SPLIT,seed=123):
    # Random train/val split with fixed seed for reproducibility
    _np.random.seed(seed); idx=_np.arange(len(y)); _np.random.shuffle(idx); cut=int(len(y)*(1-val_split))
    tr,va=idx[:cut],idx[cut:]; return X[tr],y[tr],M[tr],X[va],y[va],M[va]
def numpy_eval_accuracy(policy,X,y,M=None):
    # Evaluate NumPy policy accuracy on 5-way action labels
    P=policy.probs(X,mask=M); preds=P.argmax(axis=1); return float((preds==y).mean())
def torch_eval_accuracy(policy,X,y,M=None,device="cpu"):
    # Evaluate Torch policy accuracy (guard if torch missing)
    if not TORCH_AVAILABLE: return 0.0
    policy.eval()
    with torch.no_grad():
        x=torch.tensor(X,dtype=torch.float32,device=device)
        m=torch.tensor(M if M is not None else _np.zeros((len(X),5)),dtype=torch.float32,device=device)
        logits,_=policy(x,mask=m); preds=logits.argmax(dim=1).cpu().numpy()
        return float((preds==y).mean())

def train_imitation_numpy(policy,epochs=IMITATION_EPOCHS,hands_per_epoch=TRAIN_HANDS_PER_EPOCH):
    # Simple one-step supervised update per epoch on synthetic data
    for ep in range(1,epochs+1):
        print(f"\n[NumPy Imitation] Generating {hands_per_epoch} hands...")
        X,y,M=generate_heuristic_dataset(hands_per_epoch)
        Xtr,ytr,Mtr,Xva,yva,Mva=split_train_val(X,y,M)
        policy.step_supervised(Xtr,ytr,lr=LEARNING_RATE)
        acc=numpy_eval_accuracy(policy,Xva,yva,Mva)
        print(f"[NumPy Imitation] Epoch {ep}/{epochs} — Val Acc: {acc*100:.1f}%")
    print("[NumPy Imitation] Done.")

def train_imitation_torch(policy,device="cpu",epochs=IMITATION_EPOCHS,hands_per_epoch=TRAIN_HANDS_PER_EPOCH,lr=TORCH_LR):
    # Torch imitation learning with CE loss and simple curves
    if not TORCH_AVAILABLE: print("Torch not available."); return
    policy.to(device); opt=optim.Adam(policy.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
    train_curve=[]; val_curve=[]
    for ep in range(1,epochs+1):
        print(f"\n[Torch Imitation] Generating {hands_per_epoch} hands...")
        X,y,M=generate_heuristic_dataset(hands_per_epoch)
        Xtr,ytr,Mtr,Xva,yva,Mva=split_train_val(X,y,M)
        policy.train()
        x=torch.tensor(Xtr,dtype=torch.float32,device=device); m=torch.tensor(Mtr,dtype=torch.float32,device=device); t=torch.tensor(ytr,dtype=torch.long,device=device)
        logits,_=policy(x,mask=m); loss=ce(logits,t)
        opt.zero_grad(); loss.backward(); opt.step()
        val_acc=torch_eval_accuracy(policy,Xva,yva,Mva,device=device)
        train_curve.append(float(loss.item())); val_curve.append(val_acc)
        print(f"[Torch Imitation] Epoch {ep}/{epochs} — Train CE: {loss.item():.3f} | Val Acc: {val_acc*100:.1f}%")
    if plt is not None:
        # Optional quick plot of training cross-entropy and validation accuracy
        plt.figure(); plt.plot(train_curve,label="Train CE"); plt.plot(val_curve,label="Val Acc"); plt.title("Torch Imitation"); plt.legend(); plt.show()
    print("[Torch Imitation] Done.")

def grid_search_torch(device="cpu"):
    # Small parameter grid over hidden size, lr, and epochs; return best model
    if not TORCH_AVAILABLE: print("Torch not available."); return None
    print("\n[Grid] Compact Torch grid search...")
    X,y,M=generate_heuristic_dataset(220)
    Xtr,ytr,Mtr,Xva,yva,Mva=split_train_val(X,y,M)
    best=(None,-1.0)
    for hidden in [64,128]:
        for lr in [1e-3,3e-3,1e-2]:
            for epochs in [2,3]:
                pol=TorchPolicy(FEATURE_DIM,hidden=hidden).to(device); opt=optim.Adam(pol.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
                for _ in range(epochs):
                    pol.train()
                    x=torch.tensor(Xtr,dtype=torch.float32,device=device); m=torch.tensor(Mtr,dtype=torch.float32,device=device); t=torch.tensor(ytr,dtype=torch.long,device=device)
                    logits,_=pol(x,mask=m); loss=ce(logits,t); opt.zero_grad(); loss.backward(); opt.step()
                acc=torch_eval_accuracy(pol,Xva,yva,Mva,device=device)
                print(f"[Grid] h={hidden} lr={lr} ep={epochs} -> ValAcc {acc*100:.1f}%")
                if acc>best[1]: best=(pol,acc)
    print("[Grid] Best Val Acc: {:.1f}%".format(best[1]*100)); return best[0]

def bayes_opt_torch(device="cpu", iters=12, init=6):
    # Bayesian-style hyperparameter search:
    # - Start with `init` random samples
    # - Then take `iters` local perturbation steps around the best
    if not TORCH_AVAILABLE: print("Torch not available."); return None
    print("\n[Bayes] Quick Bayesian-style search (random init + local refine)...")
    # Generate a small imitation dataset once and reuse for speed
    X,y,M=generate_heuristic_dataset(240)
    Xtr,ytr,Mtr,Xva,yva,Mva=split_train_val(X,y,M)
    tried=[]; best=(None,-1.0,(0,0.0))  # (best_model, best_acc, (h, lr, ep))
    def eval_cfg(h,lr,ep):
        # Train a small classifier with given hidden size, lr, and epochs
        pol=TorchPolicy(FEATURE_DIM,hidden=h).to(device); opt=optim.Adam(pol.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
        for _ in range(ep):
            pol.train()
            x=torch.tensor(Xtr,dtype=torch.float32,device=device); m=torch.tensor(Mtr,dtype=torch.float32,device=device); t=torch.tensor(ytr,dtype=torch.long,device=device)
            logits,_=pol(x,mask=m); loss=ce(logits,t)
            opt.zero_grad(); loss.backward(); opt.step()
        # Validate
        acc=torch_eval_accuracy(pol,Xva,yva,Mva,device=device)
        return pol, acc
    # Random initializations
    for k in range(init):
        h=random.choice([64,96,128,160]); lr=10**random.uniform(-3.5,-2.5); ep=random.choice([2,3,4])
        pol,acc=eval_cfg(h,lr,ep); tried.append((h,lr,ep,acc))
        if acc>best[1]: best=(pol,acc,(h,lr,ep))
        print(f"[Bayes] init {k+1}/{init}: h={h} lr={lr:.4f} ep={ep} -> {acc*100:.1f}%")
    # Local refinement around current best (Gaussian perturbations with clamps)
    for t in range(iters):
        bh,blr,bep=best[2]
        h=int(max(32, min(192, int(random.gauss(bh,20)))))          # jitter hidden size
        lr=max(5e-4, min(2e-2, random.gauss(blr, blr*0.3)))         # jitter learning rate
        ep=int(max(2, min(5, int(random.gauss(bep,0.8)))))          # jitter epochs
        pol,acc=eval_cfg(h,lr,ep); tried.append((h,lr,ep,acc))
        if acc>best[1]: best=(pol,acc,(h,lr,ep))
        print(f"[Bayes] step {t+1}/{iters}: h={h} lr={lr:.4f} ep={ep} -> {acc*100:.1f}% (best {best[1]*100:.1f}%)")
    print("[Bayes] Best: h={} lr={:.4f} ep={} val_acc={:.1f}%".format(best[2][0],best[2][1],best[2][2],best[1]*100))
    return best[0]

def selfplay_train_torch(episodes=SELFPLAY_EPISODES,device="cpu",log_csv=False):
    # Mini self-play loop: A2C vs Heuristic for `episodes` single-hand bouts.
    # Optionally logs per-episode rewards to CSV.
    if not TORCH_AVAILABLE: print("Torch not available."); return None
    pol=TorchPolicy(FEATURE_DIM,hidden=TORCH_HIDDEN).to(device)
    ai1=TorchA2CAgent("A2C",pol); ai2=HeuristicAI("Heuristic")
    tmp=TexasHoldemGame("Learner", ai_mode="heuristic")
    TOURNEY_LOGS.clear()
    for ep in range(1,episodes+1):
        # Alternate seats each episode to avoid position bias
        tmp.players=[ai2,ai1] if ep%2==0 else [ai1,ai2]; tmp.button=0; tmp.hand_no=0
        res=tmp.play_hand(advice=False)
        # PATCHED: log rewards by agent identity, not seat
        rew_a2c = ai1.stack - STARTING_STACK      # A2C's stack delta this episode
        rew_heur = ai2.stack - STARTING_STACK     # Heuristic's stack delta
        # Determine winner label in a seat-agnostic way
        if res["winner"] is None:
            winner_name = "Tie"
        else:
            winner_name = "A2C" if tmp.players[res["winner"]] is ai1 else "Heuristic"
        # Append a row to the tourney log
        TOURNEY_LOGS.append(dict(
            episode=ep,
            a2c_reward=rew_a2c,
            heuristic_reward=rew_heur,
            pot=res["pot"],
            winner=winner_name
        ))
        # Periodic progress print
        if ep%50==0: print(f"[Self-Play] Episode {ep}/{episodes} | A2C cum {sum(t['a2c_reward'] for t in TOURNEY_LOGS):+}")
    # Optional CSV export
    if log_csv and TOURNEY_LOGS:
        try:
            if pd is not None: pd.DataFrame(TOURNEY_LOGS).to_csv(SAVE_TOURNEY_CSV,index=False,encoding="utf-8-sig")
            else:
                import csv
                keys=list(TOURNEY_LOGS[0].keys())
                with open(SAVE_TOURNEY_CSV,"w",newline="",encoding="utf-8-sig") as f:
                    w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
                    for row in TOURNEY_LOGS: w.writerow(row)
            print(f"Saved tourney log to {os.path.abspath(SAVE_TOURNEY_CSV)}")
        except Exception as e:
            print(f"Tourney CSV save failed: {e}")
    return pol

# ---- CLI ----
def main():
    # Entry point: interactive menu -> choose mode -> optional training -> play session
    print("\nWelcome to Heads-Up Texas Hold'em!")
    if not USE_UNICODE_SUITS: print("Note: Using ASCII suits (As, Kh, Qd, Jc).")
    print("\nChoose mode:")
    print("  1) Play vs Heuristic AI (fast)")
    print("  2) Play vs Learning AI (NumPy RL)")
    print("  3) Train NumPy (imitation) then play vs trained policy")
    max_opt=3
    if TORCH_AVAILABLE:
        print("  4) Play vs Torch A2C (actor-critic)")
        print("  5) Train Torch (imitation) then play vs trained policy")
        print("  6) Torch self-play tourney (loggable) then play")
        print("  7) Torch grid search (quick) then play best")
        print("  8) Torch Bayesian-style search (quick) then play best")
        max_opt=8
    choice=input_int("Select",1,max_opt)

    # Defaults and device selection
    ai_mode="heuristic"; device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    numpy_policy=None; learned_torch=None

    # Route chosen mode
    if choice==3:
        # Train NumPy imitation model then use RLAgent wrapper
        numpy_policy=MLPPolicy(FEATURE_DIM,hidden=64,seed=123); train_imitation_numpy(numpy_policy,IMITATION_EPOCHS,TRAIN_HANDS_PER_EPOCH); ai_mode="rl_numpy"
    elif choice==2:
        # Use an untrained NumPy policy (learns online via REINFORCE)
        numpy_policy=MLPPolicy(FEATURE_DIM,hidden=64,seed=42); ai_mode="rl_numpy"
    elif TORCH_AVAILABLE and choice==4:
        # Play against a fresh Torch A2C policy (learns online)
        ai_mode="rl_torch"; learned_torch=TorchPolicy(FEATURE_DIM,hidden=TORCH_HIDDEN).to(device)
    elif TORCH_AVAILABLE and choice==5:
        # Train Torch imitation model, then play with it
        learned_torch=TorchPolicy(FEATURE_DIM,hidden=TORCH_HIDDEN).to(device); train_imitation_torch(learned_torch,device,IMITATION_EPOCHS,TRAIN_HANDS_PER_EPOCH,TORCH_LR); ai_mode="rl_torch"
    elif TORCH_AVAILABLE and choice==6:
        # Self-play bootstrap (A2C vs Heuristic), optionally log CSV
        log=input_yes_no("Save long tourney self-play log to CSV after training?"); learned_torch=selfplay_train_torch(episodes=SELFPLAY_EPISODES,device=device,log_csv=log); ai_mode="rl_torch"
    elif TORCH_AVAILABLE and choice==7:
        # Tiny grid search over Torch hyperparameters
        learned_torch=grid_search_torch(device=device); ai_mode="rl_torch"
    elif TORCH_AVAILABLE and choice==8:
        # Bayesian-style search (random init + refine)
        learned_torch=bayes_opt_torch(device=device,iters=10,init=6); ai_mode="rl_torch"

    # Gather session options
    user_name=input("What's your name? ").strip() or "You"
    if input_yes_no("Do you want a rules walkthrough for new players?"): print_rules()
    print("\nAbout session length:\n  • Enter how many COMPLETE hands you want to play in this session.\n  • Chips carry over; session ends early if someone busts.")
    advice=input_yes_no("\nEnable AI-Powered Advice during your turns?")
    hands=input_int("How many hands would you like to play",1,1000)

    # Build and run session
    sess=Session(user_name=user_name,hands=hands,stacks=(STARTING_STACK,STARTING_STACK),ai_mode=ai_mode,torch_device=device,torch_policy=learned_torch)
    if ai_mode=="rl_numpy" and isinstance(sess.game.players[1],RLAgent) and numpy_policy is not None:
        # Plug the trained NumPy policy into the RL agent if we trained one
        sess.game.players[1].policy=numpy_policy
    sess.play(advice=advice)

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print("\n\nExiting. Goodbye!")
        sys.exit(0)
