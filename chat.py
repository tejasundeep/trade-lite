import os
from dotenv import load_dotenv
from engine.chatbot import TradingChatbot
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

load_dotenv()

def main():
    console = Console()
    bot = TradingChatbot()
    
    console.print(Panel.fit(
        "[bold blue]TradeLite AI Assistant[/bold blue]\n"
        "Ask me about current trades, history, or strategy reasoning.",
        border_style="blue"
    ))
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        console.print("[yellow]Warning: DEEPSEEK_API_KEY not found in .env. AI features will be disabled.[/yellow]")

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            with console.status("[bold green]Analyzing market data and generating explanation...[/bold green]"):
                answer = bot.ask(query)
            
            console.print("\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(answer))
            console.print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
