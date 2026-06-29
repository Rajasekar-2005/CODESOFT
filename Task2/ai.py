"""
AI Player using Minimax algorithm with Alpha-Beta Pruning
"""

from player import Player
import math

class AIPlayer(Player):
    def __init__(self, symbol, name="AI", difficulty="hard"):
        """
        Initialize AI player
        difficulty: 'easy', 'medium', 'hard'
        """
        super().__init__(symbol, name)
        self.difficulty = difficulty
        self.nodes_evaluated = 0
        
        # Determine opponent symbol
        self.opponent = 'O' if symbol == 'X' else 'X'
    
    def get_move(self, board):
        """Get best move using Minimax algorithm"""
        print(f"\n{self.name} is thinking...")
        self.nodes_evaluated = 0
        
        if self.difficulty == 'easy':
            move = self.get_random_move(board)
        elif self.difficulty == 'medium':
            # 50% chance of best move, 50% random
            import random
            if random.random() < 0.5:
                move = self.get_best_move(board)
            else:
                move = self.get_random_move(board)
        else:  # hard
            move = self.get_best_move(board)
        
        print(f"AI evaluated {self.nodes_evaluated} possible game states.")
        return move
    
    def get_random_move(self, board):
        """Get random valid move"""
        import random
        available_moves = board.get_available_moves()
        return random.choice(available_moves) if available_moves else None
    
    def get_best_move(self, board):
        """Get best move using Minimax with Alpha-Beta Pruning"""
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        
        for move in board.get_available_moves():
            # Try this move
            board.make_move(move, self.symbol)
            
            # Calculate score using minimax
            score = self.minimax(board, 0, False, alpha, beta)
            
            # Undo move
            board.undo_move(move)
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        return best_move
    
    def minimax(self, board, depth, is_maximizing, alpha, beta):
        """
        Minimax algorithm with Alpha-Beta Pruning
        
        Args:
            board: Current board state
            depth: Current depth in game tree
            is_maximizing: True if maximizing player's turn
            alpha: Alpha value for pruning
            beta: Beta value for pruning
        
        Returns:
            Best score for current position
        """
        self.nodes_evaluated += 1
        
        # Check terminal states
        if board.check_winner(self.symbol):
            return 10 - depth  # Prefer faster wins
        elif board.check_winner(self.opponent):
            return depth - 10  # Prefer slower losses
        elif board.is_board_full():
            return 0  # Draw
        
        if is_maximizing:
            # Maximizing player (AI)
            max_score = -math.inf
            
            for move in board.get_available_moves():
                board.make_move(move, self.symbol)
                score = self.minimax(board, depth + 1, False, alpha, beta)
                board.undo_move(move)
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                # Alpha-Beta Pruning
                if beta <= alpha:
                    break
            
            return max_score
        else:
            # Minimizing player (Opponent)
            min_score = math.inf
            
            for move in board.get_available_moves():
                board.make_move(move, self.opponent)
                score = self.minimax(board, depth + 1, True, alpha, beta)
                board.undo_move(move)
                
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                # Alpha-Beta Pruning
                if beta <= alpha:
                    break
            
            return min_score
    
    def get_move_explanation(self, board, move):
        """Explain why AI chose this move"""
        explanations = {
            4: "Center is the strongest position!",
            0: "Corner positions are strategically strong!",
            2: "Corner positions are strategically strong!",
            6: "Corner positions are strategically strong!",
            8: "Corner positions are strategically strong!",
        }
        return explanations.get(move, "This is my best calculated move!")