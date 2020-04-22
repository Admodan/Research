import os


dir = r'C:\Users\adamyarschenko\Documents\Research\Research_GitHub\Python Scripts'
os.chdir(dir)

#Example of using DICOM keys to access values.
'''
Target max dose is a sub field under "Dose Reference Sequence" in the dicom header.
It can be accessed using the names, or using the index keys preceeded with '0x' which
tells python it's a hexadecimal.
What it looks like in the header. Which you can see by typing 'ds' after you've loaded the file.
(300a, 0010)  Dose Reference Sequence   1 item(s) ---- 
   (300a, 0012) Dose Reference Number               IS: "1"
   (300a, 0013) Dose Reference UID                  UI: 1.2.246.352.71.10.747826906949.850181.20190131153123
   (300a, 0014) Dose Reference Structure Type       CS: 'SITE'
   (300a, 0016) Dose Reference Description          LO: 'LtBppt'
   (300a, 0020) Dose Reference Type                 CS: 'TARGET'
   (300a, 0023) Delivery Maximum Dose               DS: "42.56"
   (300a, 0026) Target Prescription Dose            DS: "42.56"
   (300a, 0027) Target Maximum Dose                 DS: "42.56"

ds.DoseReferenceSequence[0].TargetMaximumDose
ds[0x300a,0x10][0][0x300a,0x27]

'''

'''
#Testing the python interactive window
x = np.linspace(0, 5, 10)
y = x ** 2
plt.plot(x, y, 'r', x, x ** 4, 'g', x, x ** 4, 'b')
plt.show()
'''

#os.path.exists(r'\\TBMPFS\Research')
#os.listdir(r'\\TBMPFS\Research\AY\Exported Plans\Plan1')

#Load a patient file.
"""
Information can be extracted from the plan (RP.*), structure set (RS.*), or the dose (RD.*) files in the DICOM set. 
"""
filepath = r'\\TBMPFS\Research\AY\Exported Plans\Plan1\RS.C039092.BreastJan2019.DIBH_LtBrt.dcm'
ds = pydicom.filereader.dcmread(filepath)

#Write contents of DICOM header to a text file. File is stored in current working directory.
with open('DICOM Contents.txt', 'w') as f:
    f.write(str(ds))









