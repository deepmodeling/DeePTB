from dptb.entrypoints.main import main as entry_main
import logging
import pyfiglet
from dptb import __version__

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def print_logo():
    f = pyfiglet.Figlet(font='dos_rebel')  # 您可以选择您喜欢的字体
    logo = f.renderText("DeePTB")
    log.info(" ")
    log.info(" ")
    log.info("#"*81)
    log.info("#" + " "*79 + "#")
    log.info("#" + " "*79 + "#")
    for line in logo.split('\n'):
        if line.strip():  # 避免记录空行
            log.info('#     '+line+ '     #')
    log.info("#" + " "*79 + "#")
    version_info = f"Version: {__version__}"
    padding = (79 - len(version_info)) // 2
    nspace = 79-padding
    format_str = "#" + "{}"+"{:<"+f"{nspace}" + "}"+ "#"         
    log.info(format_str.format(" "*padding, version_info))
    log.info("#" + " "*79 + "#")
    log.info("#"*81)
    log.info(" ")
    log.info(" ")
def main() -> None:
    """
    The main entry point for the dptb package.
    """
    print_logo()
    entry_main()

if __name__ == '__main__':
    #print_logo()
    main()
