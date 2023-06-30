import numpy as np
from StructBio.Scenographics.solids import Solids
from StructBio.Utilities.miscFunctions import colorByName
from StructBio.Utilities.miscFunctions import residueKDhColor
from StructBio.Utilities.shellNeighbors import Shell

class ResidueCenter:
    def __init__(self, resObj):
        self.residue = resObj

    def coord(self):
        res = self.residue
        cbExists = res.findAtom('CB')
        if not cbExists:
            coords = np.array(res.findAtom('CA').coord())
        else:
            coords = np.array([0.0,0.0,0.0])
            counter = 0
            for atom in res.atoms:
                if atom.name not in ['CA','C','N','O']:
                    coords += np.array(atom.coord())
                    counter += 1
            coords = coords/counter
        return coords


phobic_spheres = Solids() #Hydrophilic residues
philic_spheres = Solids() #Hydrophobic residues
HH = Solids() #hydrophobic-hydrophobic
HP = Solids() #hydrophobic - hydrophilic
PP = Solids() #hydrophilic-hydrophilic


pdbID = raw_input("Type in four character PDB Identifier: \n")
p = chimera.openModels.open(pdbID, type = "PDB")[0]

resDistThreshold = 8.0
res_centroids = []
for R in p.residues:
    if not R.kdHydrophobicity:
        continue
    if R.findAtom('CA'):
        cen = ResidueCenter(R)
        res_centroids.append(cen)
        if R.kdHydrophobicity < 0:
            philic_spheres.addSphere(cen.coord(), 0.6, residueKDhColor(R))
        if R.kdHydrophobicity > 0:
            phobic_spheres.addSphere(cen.coord(), 0.6, residueKDhColor(R))
        
rCshell = Shell(res_centroids, 0.0, resDistThreshold)


for res in res_centroids:
    for neighbor in rCshell.getAtomsInShell(res):
        if neighbor.residue.id.chainId < res.residue.id.chainId:
            continue
        if (neighbor.residue.id.chainId == res.residue.id.chainId and
            neighbor.residue.id.position < res.residue.id.position):
            continue
        kdColor = tuple(0.5*( np.array(residueKDhColor(res.residue)) +  np.array(residueKDhColor(neighbor.residue)) ) )
        
        if res.residue.kdHydrophobicity < 0 and neighbor.residue.kdHydrophobicity < 0: #both hydrophilic pairs, PP
            PP.addTube(res.coord(), neighbor.coord(), 0.3, kdColor)
        if res.residue.kdHydrophobicity >= 0 and neighbor.residue.kdHydrophobicity >= 0: #both hydrophobic pairs, HH
            HH.addTube(res.coord(), neighbor.coord(), 0.3, kdColor)
        if neighbor.residue.kdHydrophobicity*res.residue.kdHydrophobicity < 0 : #HP pair
            HP.addTube(res.coord(), neighbor.coord(), 0.3, kdColor)

            
        
        
phobic_spheres.display()
philic_spheres.display()
HH.display()
HP.display()
PP.display()
