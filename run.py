#!/usr/bin/env python3
"""
Simple runner script for the chatbot
"""

from chatbot import RuleBasedChatbot

if __name__ == "__main__":
    print("Starting the chatbot...")
    bot = RuleBasedChatbot()
    bot.start()