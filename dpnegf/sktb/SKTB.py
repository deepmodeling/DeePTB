import time
import argparse

from sktb.Parameters import Paras
from sktb.NEGF import NEGFCal
from sktb.ElectronBase import Electron

def main():
    # interact command line.
    parser = argparse.ArgumentParser(
        description="Parameters.")
    parser.add_argument('-in', '--input_file', type=str,
                        default='input.json', help='json file for inputs')
    args = parser.parse_args()
    fp = open(args.input_file)
    paras = Paras(fp)
    if paras.Task.lower() == 'band':
        elec = Electron(paras)
        Efermi = elec.GetFermi()
        Emin = paras.BandPlotRang[0]
        Emax = paras.BandPlotRang[1]
        elec.BandPlot(FigName= 'band.png',Emax=Emax,Emin=Emin,Efermi= Efermi, \
            GUI=paras.GUI, restart=False)
            
    if paras.Task.lower() == 'negf':
        tst = time.time()
        negf = NEGFCal(paras)
        ted = time.time()
        print("# initialization: %12.2f sec." %(ted-tst))
        
        if paras.ShowContactBand in paras.Contacts:
            negf.CalContBand(tag = paras.ShowContactBand)
    
        if not paras.ContactsOnly:
            negf.CalTrans()

        

if __name__ == "__main__":
    main()
