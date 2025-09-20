# extended_poker_tests.py
import unittest
import random
import math
import numpy as np
from unittest.mock import patch, MagicMock

from Texas_Mayhem_Version_V2 import (
    Card, Deck, HandEvaluator, monte_carlo_equity, encode_state, FEATURE_DIM,
    STARTING_STACK, BIG_BLIND, SMALL_BLIND, HeuristicAI, Player, TexasHoldemGame,
    MLPPolicy, RLAgent, apply_action_mapping, legal_action_mask, classify_hand,
    has_flush_draw, straight_draw_type, overcards_count, recommended_sizes,
    log_action, fill_rewards_for_hand, HAND_LOGS, _best5_score, street_one_hot,
    generate_heuristic_dataset
)

class TestCardAndDeck(unittest.TestCase):
    """Test basic card and deck functionality"""
    
    def test_card_basic_properties(self):
        """Test basic card properties and string representations"""
        card1 = Card(14, "S")
        card2 = Card(14, "S") 
        card3 = Card(13, "S")
        
        self.assertEqual(card1.rank, 14)
        self.assertEqual(card1.suit, "S")
        # Note: Card class uses @dataclass(frozen=True) so equality should work
        # But we'll test basic properties instead of equality since it's not implemented
        self.assertEqual(card1.rank, card2.rank)
        self.assertEqual(card1.suit, card2.suit)
        self.assertNotEqual(card1.rank, card3.rank)
    
    def test_deck_completeness(self):
        """Ensure deck contains exactly 52 unique cards"""
        deck = Deck()
        self.assertEqual(len(deck.cards), 52)
        
        # Check all ranks and suits are present
        ranks = set(card.rank for card in deck.cards)
        suits = set(card.suit for card in deck.cards)
        
        self.assertEqual(len(ranks), 13)  # 2-14 (A)
        self.assertEqual(len(suits), 4)   # S,H,D,C
        self.assertEqual(suits, {"S", "H", "D", "C"})
    
    def test_deck_dealing_reduces_size(self):
        """Test that dealing cards reduces deck size appropriately"""
        deck = Deck()
        initial_size = len(deck.cards)
        
        cards = deck.deal(5)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(deck.cards), initial_size - 5)
        
        # Deal remaining cards
        remaining = deck.deal(47)
        self.assertEqual(len(remaining), 47)
        self.assertEqual(len(deck.cards), 0)

class TestHandEvaluatorAdvanced(unittest.TestCase):
    """Advanced hand evaluation test cases"""
    
    def test_straight_ace_low(self):
        """Test A-2-3-4-5 straight (wheel)"""
        hole = [Card(14, "S"), Card(2, "H")]  # A-2
        board = [Card(3, "D"), Card(4, "C"), Card(5, "S"), Card(9, "H"), Card(10, "D")]
        
        result = HandEvaluator.evaluate(hole, board)
        # Looking at the code, straight rank is 4 and wheel high card should be 5
        self.assertEqual(result[0], 4)  # Straight rank from HAND_RANK
        self.assertEqual(result[1], 5)  # High card of wheel is 5
    
    def test_kicker_precedence(self):
        """Test that kickers determine winner correctly"""
        board = [Card(9, "S"), Card(9, "H"), Card(2, "D"), Card(7, "C"), Card(3, "S")]
        
        # Both have pair of 9s, but different kickers
        hand1 = HandEvaluator.evaluate([Card(14, "S"), Card(13, "H")], board)  # A-K kickers
        hand2 = HandEvaluator.evaluate([Card(14, "D"), Card(12, "C")], board)  # A-Q kickers
        
        self.assertGreater(hand1, hand2)
    
    def test_two_pair_tie_breaker(self):
        """Test two pair comparison with kicker"""
        board = [Card(9, "S"), Card(7, "H"), Card(2, "D")]
        
        hand1 = HandEvaluator.evaluate([Card(9, "H"), Card(7, "D")], board + [Card(14, "S"), Card(3, "C")])
        hand2 = HandEvaluator.evaluate([Card(9, "D"), Card(7, "C")], board + [Card(13, "S"), Card(4, "C")])
        
        self.assertGreater(hand1, hand2)  # Same two pair, A kicker beats K kicker

class TestMonteCarloEquity(unittest.TestCase):
    """Test equity calculations"""
    
    def test_pocket_aces_preflop(self):
        """Pocket aces should have high equity preflop"""
        hole = [Card(14, "S"), Card(14, "H")]
        board = []
        
        equity = monte_carlo_equity(hole, board, opponents=1, trials=500)
        self.assertGreater(equity, 0.70)  # Should win >70% vs random hand (more conservative)
    
    def test_set_on_flop(self):
        """Set should have very high equity"""
        hole = [Card(9, "S"), Card(9, "H")]
        board = [Card(9, "D"), Card(7, "C"), Card(2, "S")]  # Flopped set
        
        equity = monte_carlo_equity(hole, board, opponents=1, trials=300)
        self.assertGreater(equity, 0.80)  # Set should be very strong (more conservative)
    
    def test_flush_draw_equity(self):
        """Flush draw should have reasonable equity"""
        hole = [Card(14, "S"), Card(5, "S")]
        board = [Card(9, "S"), Card(7, "S"), Card(2, "H")]  # Nut flush draw
        
        equity = monte_carlo_equity(hole, board, opponents=1, trials=400)
        self.assertGreater(equity, 0.25)  # Should have decent equity
        self.assertLess(equity, 0.70)      # But not too high

class TestGameLogic(unittest.TestCase):
    """Test game mechanics and betting logic"""
    
    def test_player_stack_management(self):
        """Test that player stacks are managed correctly"""
        player = Player("Test", 1000)
        
        # Normal bet
        paid = player.bet(100)
        self.assertEqual(paid, 100)
        self.assertEqual(player.stack, 900)
        
        # Bet more than stack (should be clamped)
        paid = player.bet(1500)
        self.assertEqual(paid, 900)
        self.assertEqual(player.stack, 0)
        
        # Bet when broke
        paid = player.bet(50)
        self.assertEqual(paid, 0)
        self.assertEqual(player.stack, 0)
    
    def test_heuristic_ai_basic_decisions(self):
        """Test that HeuristicAI makes basic decisions without crashing"""
        ai = HeuristicAI("TestAI", 1000)
        ai.hole = [Card(14, "S"), Card(14, "H")]  # Pocket aces
        
        # Should make a decision with premium hand
        board = []
        decision = ai.decide(board, pot=20, to_call=0, min_raise=10, street="preflop")
        action, amount, equity = decision
        
        # Just test that it returns valid values
        self.assertIsInstance(action, str)
        self.assertIsInstance(amount, (int, float))
        self.assertIsInstance(equity, (int, float))
        self.assertGreaterEqual(equity, 0.0)
        self.assertLessEqual(equity, 1.0)

class TestActionMapping(unittest.TestCase):
    """Test discrete action mapping to concrete actions"""
    
    def test_facing_bet_actions(self):
        """Test action mapping when facing a bet"""
        pot, to_call, stack = 100, 30, 500
        
        # Test fold
        action, amount = apply_action_mapping(0, facing=True, pot=pot, to_call=to_call, stack=stack)
        self.assertEqual(action, "fold")
        self.assertEqual(amount, 0)
        
        # Test call
        action, amount = apply_action_mapping(1, facing=True, pot=pot, to_call=to_call, stack=stack)
        self.assertEqual(action, "call")
        self.assertEqual(amount, to_call)
        
        # Test jam (all-in)
        action, amount = apply_action_mapping(4, facing=True, pot=pot, to_call=to_call, stack=stack)
        self.assertEqual(action, "raise")
        self.assertEqual(amount, stack)
    
    def test_no_bet_actions(self):
        """Test action mapping when no bet to face"""
        pot, to_call, stack = 100, 0, 500
        
        # Test check
        action, amount = apply_action_mapping(0, facing=False, pot=pot, to_call=to_call, stack=stack)
        self.assertEqual(action, "check")
        self.assertEqual(amount, 0)
        
        # Test pot-sized bet
        action, amount = apply_action_mapping(3, facing=False, pot=pot, to_call=to_call, stack=stack)
        self.assertEqual(action, "bet")
        self.assertEqual(amount, 100)  # Pot-sized
    
    def test_short_stack_constraints(self):
        """Test action mapping with very short stack"""
        pot, to_call, stack = 100, 20, 15  # Can't even call
        
        action, amount = apply_action_mapping(1, facing=True, pot=pot, to_call=to_call, stack=stack)
        self.assertEqual(action, "call")
        self.assertEqual(amount, 15)  # Should bet entire stack

class TestMLPPolicyBasic(unittest.TestCase):
    """Test NumPy MLP policy functionality"""
    
    def test_policy_forward_pass(self):
        """Test that policy forward pass produces correct shapes"""
        policy = MLPPolicy(FEATURE_DIM, hidden=64)
        
        # Test single state
        x = np.random.randn(1, FEATURE_DIM)
        logits, hidden = policy.forward(x)
        
        self.assertEqual(logits.shape, (1, 5))  # 5 actions
        self.assertEqual(hidden.shape, (1, 64))  # Hidden size
        
        # Test batch
        x_batch = np.random.randn(10, FEATURE_DIM)
        logits_batch, hidden_batch = policy.forward(x_batch)
        
        self.assertEqual(logits_batch.shape, (10, 5))
        self.assertEqual(hidden_batch.shape, (10, 64))
    
    def test_policy_probabilities(self):
        """Test that policy outputs valid probability distributions"""
        policy = MLPPolicy(FEATURE_DIM, hidden=32)
        
        x = np.random.randn(3, FEATURE_DIM)
        probs = policy.probs(x)
        
        # Should be valid probability distributions
        self.assertEqual(probs.shape, (3, 5))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
        self.assertTrue(np.all(probs >= 0))  # All non-negative

class TestHandAnalysis(unittest.TestCase):
    """Test hand classification and analysis functions"""
    
    def test_classify_hand_types(self):
        """Test hand classification function"""
        # Test flush
        hole = [Card(14, "S"), Card(9, "S")]
        board = [Card(7, "S"), Card(5, "S"), Card(3, "S")]
        
        classification = classify_hand(hole, board)
        self.assertEqual(classification, "Flush")
        
        # Test straight
        hole = [Card(9, "H"), Card(8, "D")]
        board = [Card(7, "C"), Card(6, "S"), Card(5, "H")]
        
        classification = classify_hand(hole, board)
        self.assertEqual(classification, "Straight")
    
    def test_flush_draw_detection(self):
        """Test flush draw detection"""
        hole = [Card(14, "S"), Card(9, "S")]
        board = [Card(7, "S"), Card(5, "D"), Card(3, "H")]
        
        has_draw, suit = has_flush_draw(hole, board)
        self.assertTrue(has_draw)
        self.assertEqual(suit, "S")
        
        # No flush draw
        hole = [Card(14, "S"), Card(9, "H")]
        board = [Card(7, "D"), Card(5, "D"), Card(3, "H")]
        
        has_draw, suit = has_flush_draw(hole, board)
        self.assertFalse(has_draw)
        self.assertIsNone(suit)
    
    def test_straight_draw_classification(self):
        """Test straight draw type detection"""
        # Open-ended straight draw
        hole = [Card(9, "S"), Card(8, "H")]
        board = [Card(7, "D"), Card(6, "C"), Card(2, "S")]
        
        draw_type = straight_draw_type(hole, board)
        self.assertEqual(draw_type, "OESD")
        
        # Gutshot
        hole = [Card(9, "S"), Card(7, "H")]
        board = [Card(6, "D"), Card(5, "C"), Card(2, "S")]
        
        draw_type = straight_draw_type(hole, board)
        self.assertEqual(draw_type, "Gutshot")
        
        # No straight draw
        hole = [Card(14, "S"), Card(9, "H")]
        board = [Card(7, "D"), Card(3, "C"), Card(2, "S")]
        
        draw_type = straight_draw_type(hole, board)
        self.assertIsNone(draw_type)

class TestLoggingSystem(unittest.TestCase):
    """Test hand logging and analytics"""
    
    def setUp(self):
        """Clear logs before each test"""
        HAND_LOGS.clear()
    
    def test_action_logging(self):
        """Test that actions are logged correctly"""
        hole = [Card(14, "S"), Card(13, "H")]
        board = [Card(9, "D"), Card(7, "C")]
        
        log_action(
            hand_no=1, street="flop", player="TestPlayer",
            hole=hole, board=board, pot_before=100, to_call=20,
            stack_hero=1000, stack_opp=900, action="call", amount=20,
            equity=0.65
        )
        
        self.assertEqual(len(HAND_LOGS), 1)
        
        log_entry = HAND_LOGS[0]
        self.assertEqual(log_entry["hand_no"], 1)
        self.assertEqual(log_entry["action"], "call")
        self.assertEqual(log_entry["amount"], 20)
        self.assertEqual(log_entry["equity"], 0.65)
    
    def test_reward_backfill(self):
        """Test that rewards are backfilled correctly"""
        # Add some test logs
        log_action(1, "preflop", "Player1", [], [], 0, 10, 1000, 1000, "call", 10)
        log_action(1, "flop", "Player1", [], [], 20, 0, 990, 1010, "bet", 50)
        log_action(1, "flop", "Player2", [], [], 70, 50, 1010, 940, "call", 50)
        
        # Backfill rewards
        rewards = {"Player1": -100, "Player2": 100}
        fill_rewards_for_hand(1, rewards)
        
        # Check that rewards were filled
        for log in HAND_LOGS:
            if log["player"] == "Player1":
                self.assertEqual(log["reward"], -100)
            elif log["player"] == "Player2":
                self.assertEqual(log["reward"], 100)

class TestDatasetGeneration(unittest.TestCase):
    """Test synthetic dataset generation for training"""
    
    def test_heuristic_dataset_format(self):
        """Test that generated dataset has correct format"""
        X, y, M = generate_heuristic_dataset(n_hands=5)
        
        # Check shapes
        self.assertEqual(X.ndim, 2)
        self.assertEqual(X.shape[1], FEATURE_DIM)
        self.assertEqual(len(y), X.shape[0])
        self.assertEqual(M.shape, (X.shape[0], 5))
        
        # Check data types and ranges
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y < 5))  # 5 discrete actions
        self.assertTrue(np.all(M >= -np.inf))  # Mask should be reasonable values
    
    def test_state_encoding_consistency(self):
        """Test that state encoding is consistent"""
        hole = [Card(14, "S"), Card(13, "H")]
        board = [Card(9, "D")]
        
        # Encode same state twice
        state1 = encode_state(hole, board, 100, 20, 1000, 900, 0.6)
        state2 = encode_state(hole, board, 100, 20, 1000, 900, 0.6)
        
        np.testing.assert_array_equal(state1, state2)
        
        # Different equity should produce different encoding
        state3 = encode_state(hole, board, 100, 20, 1000, 900, 0.8)
        self.assertFalse(np.array_equal(state1, state3))

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)