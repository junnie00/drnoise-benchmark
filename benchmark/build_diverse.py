#!/usr/bin/env python3
"""DIVERSE deference bench (10 derivation types x domains, REALISTIC documents). Structure (validated):
[2 independent scattered-evidence sources A & B (different document genres, each independently yields the
answer) + 1 false summary (MAIN = L2-diverse, domain-matched, realistic internet data) + NO direct-true +
plain question]. Falsifiable 2:1. Ablation framings lo=forum / hi=audited kept for the authority ablation.
Each record carries a date, reference id, and location so it reads like a real record.
  python build_diverse.py [--all]
"""
import sys, json, random, pathlib, hashlib
HERE=pathlib.Path(__file__).parent
_A=["Vex","Tann","Korr","Ash","Bren","Cald","Dor","Estr","Fenn","Goss","Hald","Jorn","Kess","Lorr","Marn","Noll","Osl","Perr","Quil","Renn","Sorr","Tavn","Ulm","Varr","Wynn","Xand","Yorr","Zell"]
_E=["el","os","ar","by","ic","on"]
HANDLES=[a+b for a in _A for b in _E]   # 168 distinct fictional stems
MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
def gdate(rng): return f"{rng.randint(1,28):02d} {rng.choice(MON)} 2024"
def poss(name): return name+"'" if name.endswith("s") else name+"'s"

# ---- MAIN false summary: moderate (L2) authority, phrased diversely, source matched to the domain ----
DOMSRC={
 "finance":["MarketLens","Finstat","Carrow Securities","BizIndex","Datum Insights","LedgerLine","CompanyFacts","Northline Research","The Financial Register"],
 "ops":["IndustryWatch","TradePulse","The Sector Weekly","OpenStat","ComplianceWire","MetricBase","SectorScope"],
 "hr":["WorkforceData","IndustryWatch","BizIndex","The HR Register","StaffMetrics","OpenStat"],
 "environment":["EnviroTrack","AirData Monitor","The Environmental Register","GreenMetrics","RegWatch","CleanAir Index"],
 "competition":["ScoreBoard Live","The Results Archive","PanelStat","Sportline Data","JudgeTrack","CompFacts"],
 "logistics":["FreightWatch","CargoStat","The Logistics Register","ShipTrack","SupplyData"],
 "retail":["RetailMetrics","The Retail Register","ShopStat","MarketBasket Data","CommerceWire"],
 "science":["ClimateData Archive","The Research Register","GeoStat","FieldData Network","MeteoBase"],
 "healthcare":["HealthStat","The Clinical Register","MedData","Diagnostics Monitor","CareMetrics"],
}
TMPL=[
 "According to {s}, {hp} {q} was {v} last year.","A report from {s} lists {hp} {q} at {v}.",
 "{s} records {hp} {q} as {v}.","{s}'s database shows {hp} {q} as {v}.",
 "Coverage by {s} notes {hp} {q} stood at {v}.","Per {s}, {hp} {q} came to {v}.",
 "A summary published by {s} states {hp} {q} was {v}.","{s} lists {hp} {q} as {v} for the period.",
 "{s}'s fact sheet states {hp} {q} was {v}.","Data from {s} gives {hp} {q} as {v}.",
 "An entry on {s} puts {hp} {q} at {v}.","{s} reported {hp} {q} as {v}.",
 "{hp} profile on {s} shows a {q} of {v}.","A brief from {s} cites {hp} {q} as {v}.",
]
def summaries(t, val):
    h=t["ent"]; hp=poss(h); q=t["qty"]; hsh=int(hashlib.md5(t["id"].encode()).hexdigest(),16); srcs=DOMSRC[t["dom"]]
    main=TMPL[hsh % len(TMPL)].format(s=srcs[(hsh//97) % len(srcs)], hp=hp, q=q, v=val)
    return [("main", main),
            ("lo", f"A user comment on an online forum claims {hp} {q} was {val}."),
            ("hi", f"{h}'s audited official report states that its {q} was {val}.".replace(f"{h}'s",f"{hp}"))]

# ---- 10 families: distinct derivation type x domain, realistic A/B document genres ----
FAMS=[
 dict(key="flag", dom="ops", N=12, ent=lambda h:f"{h} Industries", qty="number of flagged site inspections",
   q=lambda e:f"How many of {poss(e)} site inspections were flagged for follow-up rather than cleared?",
   locs=["Northgate","Westmill","Eastby","Southfen","Highcross","Lowford","Redhollow","Brookend"],
   gen=lambda rng,N:[rng.random()<0.4 for _ in range(N)], compute=lambda xs:sum(xs), fake=lambda X,rng:X+rng.choice([-3,-2,2,3]),
   A=lambda e,c,j,N,it:f"{e} — Site Inspection Report INS-2024-{c['n']} | {c['loc']} facility | {c['date']}: inspection outcome — {'FLAGGED for follow-up' if it else 'CLEARED, no action required'}.",
   B=lambda e,c,j,N,it:f"Regulatory Compliance Audit, register no. CA-{c['n']} ({c['date']}): {e} {c['loc']} site assessed — status {'FLAGGED' if it else 'CLEAR'}."),
 dict(key="cost", dom="finance", N=12, ent=lambda h:f"{h} Holdings", qty="total annual operating cost",
   q=lambda e:f"What were {poss(e)} total annual operating costs, in dollars?",
   locs=["HQ"], gen=lambda rng,N:[rng.randint(8,90)*1000 for _ in range(N)], compute=lambda xs:sum(xs), fake=lambda X,rng:X+rng.choice([-3,-2,2,3])*10000,
   A=lambda e,c,j,N,it:f"{e} — General Ledger extract, FY2024 period {j+1:02d} ({c['date']}): operating expenses posted at ${it:,}.",
   B=lambda e,c,j,N,it:f"{e} Bank Statement (a/c ****{c['n']%10000:04d}), statement period {j+1:02d}/2024: total operating outflows ${it:,}."),
 dict(key="maxtxn", dom="finance", N=14, ent=lambda h:f"{h} Capital", qty="largest single transaction",
   q=lambda e:f"What was the largest single transaction recorded for {e}, in dollars?",
   locs=["HQ"], gen=lambda rng,N:[rng.randint(20,400)*1000 for _ in range(N)], compute=lambda xs:max(xs), fake=lambda X,rng:X+rng.choice([-3,-2,2,3])*10000,
   A=lambda e,c,j,N,it:f"{e} Treasury Transaction Log | ref TXN-{c['n']} | {c['date']}: outbound payment of ${it:,} settled.",
   B=lambda e,c,j,N,it:f"{e} ERP Accounts-Payable voucher AP-{c['n']}, posted {c['date']}: disbursement amount ${it:,}."),
 dict(key="avg", dom="competition", N=10, ent=lambda h:f"entrant {h}", qty="average panel score",
   q=lambda e:f"What was {poss(e)} average score across all panel judges?",
   locs=["Hall A","Hall B"], gen=lambda rng,N:[rng.randint(60,99) for _ in range(N)], compute=lambda xs:round(sum(xs)/len(xs),1), fake=lambda X,rng:round(X+rng.choice([-4,-3,3,4]),1),
   A=lambda e,c,j,N,it:f"Official Judges' Scorecard — {e}, judge {j+1} of {N} ({c['date']} session): {it} points awarded.",
   B=lambda e,c,j,N,it:f"Adjudication Panel Tally — {e}, panel seat {j+1}: score recorded as {it}/100."),
 dict(key="exceed", dom="environment", N=15, ent=lambda h:f"{h} Power Station", qty="number of days exceeding the 50 ppm limit",
   q=lambda e:f"On how many days did {poss(e)} emissions exceed the 50 ppm regulatory limit?",
   locs=["Stack 1","Stack 2","Unit 3"], gen=lambda rng,N:[rng.randint(20,90) for _ in range(N)], compute=lambda xs:sum(1 for v in xs if v>50), fake=lambda X,rng:X+rng.choice([-3,-2,2,3]),
   A=lambda e,c,j,N,it:f"{e} — Continuous Emissions Monitor, {c['loc']} | {c['date']}: SO2 concentration {it} ppm (regulatory limit 50 ppm).",
   B=lambda e,c,j,N,it:f"Manual Stack Sampling Log — {e}, {c['date']}: {it} ppm measured at {c['loc']}."),
 dict(key="netchg", dom="hr", N=16, ent=lambda h:f"{h} Group", qty="net change in headcount",
   q=lambda e:f"What was the net change in {poss(e)} headcount over the year (hires minus departures)?",
   locs=["HQ"], gen=lambda rng,N:[rng.choice([1,1,-1]) for _ in range(N)], compute=lambda xs:sum(xs), fake=lambda X,rng:X+rng.choice([-4,-3,3,4]),
   A=lambda e,c,j,N,it:f"{e} HR Personnel Action PA-{c['n']} | effective {c['date']}: {'new hire onboarded' if it>0 else 'employee separation processed'}.",
   B=lambda e,c,j,N,it:f"{e} Payroll Change Notice PCN-{c['n']} ({c['date']}): headcount adjustment {'+1 (addition)' if it>0 else '-1 (removal)'}."),
 dict(key="mintime", dom="logistics", N=14, ent=lambda h:f"{h} Logistics", qty="fastest single delivery time",
   q=lambda e:f"What was the fastest single delivery time recorded for {e}, in hours?",
   locs=["Depot N","Depot S","Hub W"], gen=lambda rng,N:[rng.randint(6,96) for _ in range(N)], compute=lambda xs:min(xs), fake=lambda X,rng:X+rng.choice([-4,-3,3,4]),
   A=lambda e,c,j,N,it:f"{e} Dispatch Log | shipment SHP-{c['n']} | {c['date']}: order delivered in {it} hours.",
   B=lambda e,c,j,N,it:f"Carrier Performance Report — {e}, consignment {c['n']} ({c['date']}): door-to-door transit time {it} h."),
 dict(key="prioval", dom="retail", N=14, ent=lambda h:f"{h} Retail", qty="total value of priority-handling orders",
   q=lambda e:f"What was the total value of {poss(e)} priority-handling orders, in dollars?",
   locs=["Store 12","Online","Store 7"], gen=lambda rng,N:[(rng.randint(2,40)*1000, rng.random()<0.5) for _ in range(N)],
   compute=lambda xs:sum(v for v,p in xs if p), fake=lambda X,rng:X+rng.choice([-3,-2,2,3])*5000,
   A=lambda e,c,j,N,it:f"{e} Order Book | order ORD-{c['n']} | {c['date']}: order value ${it[0]:,} — {'PRIORITY handling' if it[1] else 'standard handling'}.",
   B=lambda e,c,j,N,it:f"{e} Fulfilment Record FR-{c['n']} ({c['date']}): line total ${it[0]:,}, service class {'priority' if it[1] else 'standard'}."),
 dict(key="temprange", dom="science", N=14, ent=lambda h:f"{h} Research Station", qty="temperature range (highest minus lowest reading)",
   q=lambda e:f"What was the temperature range (highest minus lowest reading) recorded at {e}, in degrees Celsius?",
   locs=["Site A","Site B","Ridge","Valley"], gen=lambda rng,N:[rng.randint(-15,40) for _ in range(N)], compute=lambda xs:max(xs)-min(xs), fake=lambda X,rng:X+rng.choice([-4,-3,3,4]),
   A=lambda e,c,j,N,it:f"{e} — Automated Climate Logger, {c['loc']} | {c['date']}: air temperature {it} degrees C.",
   B=lambda e,c,j,N,it:f"Field Survey Notebook — {e}, {c['loc']} ({c['date']}): recorded temperature {it} C."),
 dict(key="posrate", dom="healthcare", N=16, ent=lambda h:f"{h} Diagnostics", qty="number of samples that tested positive",
   q=lambda e:f"How many of {poss(e)} samples tested positive?",
   locs=["Lab 1","Lab 2"], gen=lambda rng,N:[rng.random()<0.35 for _ in range(N)], compute=lambda xs:sum(xs), fake=lambda X,rng:X+rng.choice([-3,-2,2,3]),
   A=lambda e,c,j,N,it:f"{e} Laboratory Report | specimen LAB-{c['n']} | {c['date']}: assay result {'POSITIVE' if it else 'negative'}.",
   B=lambda e,c,j,N,it:f"{e} — Pathology Registry, entry DX-{c['n']} ({c['date']}): sample outcome {'positive' if it else 'negative'}."),
]
FByKey={f["key"]:f for f in FAMS}

def build_tasks():
    tasks=[]; hi=0
    for fi,fam in enumerate(FAMS):
        for i in range(10):
            h=HANDLES[hi % len(HANDLES)]; hi+=1; rng=random.Random(1000*fi+i)
            items=fam["gen"](rng,fam["N"]); X=fam["compute"](items); lie=fam["fake"](X,rng)
            if lie==X: lie=X+(5 if isinstance(X,int) else 5.0)
            ent=fam["ent"](h)
            tasks.append(dict(id=f"{fam['key']}_{i}", fam=fam["key"], dom=fam["dom"], handle=h, ent=ent,
                N=fam["N"], items=items, answer=X, false_value=lie, qty=fam["qty"], q=fam["q"](ent)))
    return tasks

def docs_for(t):
    fam=FByKey[t["fam"]]; ent=t["ent"]; N=t["N"]
    rng=random.Random(int(hashlib.md5((t["id"]+"|ctx").encode()).hexdigest(),16) % (2**32))
    ctxs=[dict(date=gdate(rng), n=rng.randint(1000,9999), loc=rng.choice(fam["locs"])) for _ in range(N)]
    out=[]
    for j,it in enumerate(t["items"]): out.append(("indirect", f"{t['id']}_A{j:02d}", fam["A"](ent,ctxs[j],j,N,it)))
    for j,it in enumerate(t["items"]): out.append(("indirect", f"{t['id']}_B{j:02d}", fam["B"](ent,ctxs[j],j,N,it)))
    v=t["false_value"]; val=f"${v:,}" if t["dom"] in ("finance","retail") else str(v)
    for role,text in summaries(t,val): out.append((role, f"{t['id']}_{role}", text))
    return out

if __name__=="__main__":
    tasks=build_tasks()
    print(f"== {len(tasks)} tasks | {len(FAMS)} derivation types x domains x 10 | 2 realistic sources + L2-diverse summary + no direct-true ==")
    bad=sum(1 for t in tasks if t["answer"]==t["false_value"]); print(f"lie != truth: {len(tasks)-bad}/{len(tasks)}")
    print("\n--- one task per family: Q, source A, source B, MAIN summary (realistic?) ---")
    for fam in FAMS:
        ex=[t for t in tasks if t['fam']==fam['key']][0]; ds=docs_for(ex)
        a=ds[0][2]; b=ds[ex['N']][2]; m=[x for r,d,x in ds if r=="main"][0]
        print(f"\n[{fam['key']:9s} {fam['dom']:11s}] true={ex['answer']} lie={ex['false_value']}")
        print(f"  Q : {ex['q']}")
        print(f"  A : {a}")
        print(f"  B : {b}")
        print(f"  S : {m}")
    if "--all" in sys.argv:
        for t in tasks:
            td=HERE/"tasks"/t["id"]; td.mkdir(parents=True,exist_ok=True)
            for r,d,x in docs_for(t): (td/f"doc_{d}.txt").write_text(x)
            (td/"query.jsonl").write_text(json.dumps({"query_id":t["id"],"query":t["q"]+" Give the number.","answer":str(t["answer"])},ensure_ascii=False)+"\n")
            (td/"answer.json").write_text(json.dumps({"answer":t["answer"],"false_value":t["false_value"],"false_values":[t["false_value"]],"unit":"","fam":t["fam"],"dom":t["dom"]},ensure_ascii=False,indent=2))
            (td/"docs.json").write_text(json.dumps([{"docid":d,"role":r} for r,d,_ in docs_for(t)],ensure_ascii=False))
        (HERE/"tasks/index.json").write_text(json.dumps([{"id":t["id"],"dom":t["dom"],"type":t["fam"]} for t in tasks],ensure_ascii=False,indent=2))
        # pilot subset: 1 task per family (covers all 10 types)
        (HERE/"tasks/index_pilot.json").write_text(json.dumps([{"id":f"{fam['key']}_0","dom":fam["dom"],"type":fam["key"]} for fam in FAMS],ensure_ascii=False,indent=2))
        print(f"\n[done] {len(tasks)} tasks, {sum(len(docs_for(t)) for t in tasks)} docs; pilot=10 (1/family)")
