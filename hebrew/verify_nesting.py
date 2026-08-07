"""Do BHSA phrases/clauses/sentences ever span a verse boundary? Checked corpus-wide."""
import os, collections
from tf.fabric import Fabric
api = Fabric(locations=[os.path.expanduser("~/text-fabric-data/github/ETCBC/bhsa/tf/2021")],
             silent="deep").load("otype verse", silent="deep")
F, L = api.F, api.L
for lvl in ("phrase", "clause", "sentence"):
    objs = list(F.otype.s(lvl))
    multi = 0
    for o in objs:
        vs = {L.u(w, "verse")[0] for w in L.d(o, "word") if L.u(w, "verse")}
        if len(vs) > 1: multi += 1
    print(f"  {lvl:<10} {len(objs):>7} objects   spanning >1 verse: {multi:>6} "
          f"({100*multi/len(objs):.2f}%)")
