from dptb.entrypoints.main import main as entry_main
import logging
import pyfiglet
from dptb import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def print_logo():
    try:
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.align import Align
        from rich.text import Text
        # Create Figlet with classic dos_rebel font
        f = pyfiglet.Figlet(font='dos_rebel')
        logo_text = f.renderText("DeePTB")
        # Build Rich components with centered version
        logo = Align.center(Text(logo_text, style="bold cyan"))
        version = Align.center(Text(f"Version: {__version__}", style="bold white"))
        panel = Panel(
            Group(logo, version),
            border_style="blue",
            padding=(1, 2),
            title="Deep Learning Tight-Binding",
            subtitle="https://github.com/deepmodeling/DeePTB"
        )
        console = Console()
        console.print(panel)
    except ImportError:
        # Fallback if rich is not installed (though it should be)
        f = pyfiglet.Figlet(font='dos_rebel')
        print(f.renderText("DeePTB"))
        print(f"Version: {__version__}")
def main() -> None:
    """
    The main entry point for the dptb package.
    """
    print_logo()
    entry_main()

if __name__ == '__main__':
    #print_logo()
    main()
