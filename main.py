#!/usr/bin/env python3
"""
Main entry point for Tic-Tac-Toe AI game
"""

from game import TicTacToe

def main():
    """Main function to start the game"""
    try:
        game = TicTacToe()
        game.start()
    except KeyboardInterrupt:
        print("\n\nGame terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please restart the game.")

if __name__ == "__main__":
    main()