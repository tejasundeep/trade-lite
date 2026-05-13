import asyncio
import sys
import os
import time

import logging
from rich.logging import RichHandler

# Create handlers
file_handler = logging.FileHandler("test.log")
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL) 

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[file_handler, console_handler],
)
log = logging.getLogger(__name__)

async def dummy_autonomous_task():
    while True:
        # print("Autonomous task running...")
        await asyncio.sleep(5)

async def dummy_chatbot_ask(query):
    print("AI Engine is analyzing...")
    await asyncio.sleep(1) # Simulate API delay
    return f"I heard you say: {query}"

import queue
import threading

input_queue = queue.Queue()

def input_thread():
    while True:
        try:
            val = input("You: ")
            if val:
                input_queue.put(val.strip())
        except EOFError:
            break
        except Exception as e:
            print(f"Input Thread Error: {e}")
            break

async def run_ai_assistant():
    print("TradeX Pro 2.0 (Threaded Mode)")
    # Start the input thread
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()
    
    while True:
        try:
            # Check for input without blocking the loop
            if not input_queue.empty():
                query = input_queue.get()
                if not query: continue
                print(f"DEBUG: Processing query: {query}")
                answer = await dummy_chatbot_ask(query)
                print(f"\nAI Assistant: {answer}\n")
            
            await asyncio.sleep(0.1) # Yield to other tasks
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    asyncio.create_task(dummy_autonomous_task())
    await run_ai_assistant()

if __name__ == "__main__":
    asyncio.run(main())
