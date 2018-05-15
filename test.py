from algo.mclust_tcomp_launcher import MclustTCompLauncher
from algo.mclust_t1_launcher import MclustT1Launcher
from algo.mss_t1_launcher import MssT1Launcher
from algo.mss_tcomp_launcher import MssTCompLauncher

#MclustTCompLauncher().run(
#    {
#      "Asystole" : ["Skewness"],
#      "Tachycardia" : ["BinaryCovariance"],
#      "Ventricular_Tachycardia" : ["BinaryCovariance", "BinaryFrequency"],
#      "Bradycardia" : ["Count2", "BinaryCovariance"],
#      "Ventricular_Flutter_Fib" : ["Count2"]
#    }
#)

#MclustT1Launcher().run(['BinaryCovariance'])

#MssT1Launcher().run(['BinaryFrequency'], ['MinMaxAxiom', 'IntegralAxiom', 'FirstDiffAxiom', 'RelativeChangeAxiom'])

MssTCompLauncher().run(
    {
      "Asystole" : ["Count3"],
      "Tachycardia" : ["AreaBinary"],
      "Ventricular_Tachycardia" : ["BinaryCovariance"],
      "Bradycardia" : ["BinaryCovariance"],
      "Ventricular_Flutter_Fib" : ["BinaryFrequency"]
    },
    ['MinMaxAxiom', 'IntegralAxiom', 'FirstDiffAxiom', 'RelativeChangeAxiom']
)

