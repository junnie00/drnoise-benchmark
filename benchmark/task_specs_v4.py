"""Expanded task specifications for ld_varied9 v4.

The original nine v3 tasks are preserved. This module adds nine new instances
for each task type, keeping the same structure: two independently sufficient
indirect evidence sets plus one direct false summary.
"""
from __future__ import annotations

import pathlib

from task_specs_v2 import TASKS as BASE_TASKS

from build_diverse import build_tasks as build_diverse_tasks, docs_for as diverse_docs_for  # noqa: E402


SUPPLIERS = [
    ("Cairn Fasteners", "Morrow Castings", "Sable Fixtures"),
    ("Holloway Circuits", "Neris Tooling", "Aster Plating"),
    ("Kepler Valves", "Damon Housings", "Ridgeway Metals"),
    ("Marlin Optics", "Osric Wireworks", "Cobalt Frames"),
    ("Vesper Seals", "Lumen Gaskets", "Harbor Molds"),
    ("Noble Bearings", "Quince Forging", "Tidal Assemblies"),
    ("Ilex Sensors", "Barton Components", "Fallow Electronics"),
    ("Palisade Coatings", "Rook Machining", "Umber Castings"),
    ("Fenwick Laminates", "Ashford Connectors", "Torc Plastics"),
]

COMPANIES = [
    ("Bellwether Alloy", "18 Dec 2024", "04 Sep 2024", "I-431", "I-488"),
    ("Canton Robotics", "09 Nov 2024", "16 Aug 2024", "I-512", "I-559"),
    ("Dovetail Paper", "27 Oct 2024", "05 Jul 2024", "I-621", "I-668"),
    ("Everside Glass", "03 Dec 2024", "21 Sep 2024", "I-704", "I-739"),
    ("Frostline Foods", "14 Nov 2024", "30 Aug 2024", "I-812", "I-866"),
    ("Grantham Textiles", "22 Dec 2024", "11 Oct 2024", "I-903", "I-949"),
    ("Harborleaf Energy", "06 Nov 2024", "17 Sep 2024", "I-1004", "I-1048"),
    ("Istria Packaging", "28 Nov 2024", "02 Sep 2024", "I-1116", "I-1161"),
    ("Juniper Ceramics", "19 Dec 2024", "24 Oct 2024", "I-1207", "I-1252"),
]

CLINICS = [
    ("Alder Clinic", "biobank amendment", "participant withdrawal summary"),
    ("Briar Clinic", "device calibration attestation", "sterile-field variance report"),
    ("Cedar Clinic", "pediatric assent renewal", "protocol deviation nil return"),
    ("Dahlia Clinic", "data-use renewal", "unplanned-visit summary"),
    ("Elm Clinic", "radiology safety addendum", "contrast-event nil report"),
    ("Fennel Clinic", "remote-consent renewal", "screen-failure summary"),
    ("Grove Clinic", "pharmacy delegation log", "temperature excursion nil report"),
    ("Hazel Clinic", "monitoring-plan renewal", "late-sample summary"),
    ("Iris Clinic", "biospecimen transfer notice", "serious-adverse-event nil report"),
]

ROUTES = [
    ("Route Atlas", "Route Beacon", "Route Cinder", "Route Drift", "Route Ember"),
    ("Route Finch", "Route Grove", "Route Harbor", "Route Ibis", "Route Junco"),
    ("Route Kilo", "Route Laurel", "Route Mica", "Route Norwood", "Route Oriel"),
    ("Route Pacer", "Route Quarry", "Route Riven", "Route Summit", "Route Thistle"),
    ("Route Umber", "Route Vale", "Route Warden", "Route Xenon", "Route Yarrow"),
    ("Route Zephyr", "Route Alder", "Route Briar", "Route Cedar", "Route Dune"),
    ("Route Ester", "Route Flume", "Route Gable", "Route Heron", "Route Inlet"),
    ("Route Jasper", "Route Kestrel", "Route Linden", "Route Moraine", "Route Nettle"),
    ("Route Osprey", "Route Pine", "Route Quartz", "Route Rowan", "Route Slate"),
]

PROJECTS = [
    ("CT-610", "Mira Solen", "Dorian Vale", "Aster Bridge"),
    ("CT-624", "Jonah Rusk", "Elian Frost", "Beacon Relay"),
    ("CT-638", "Selene March", "Corin Pike", "Cinder Array"),
    ("CT-652", "Talia Venn", "Marek Stone", "Drift Sensor"),
    ("CT-666", "Rhea Calder", "Nolan Pryce", "Ember Works"),
    ("CT-680", "Iona Locke", "Gareth Wren", "Finch Grid"),
    ("CT-694", "Petra Moss", "Lucan Shore", "Grove Portal"),
    ("CT-708", "Vera Holt", "Simon Keir", "Harbor Index"),
    ("CT-722", "Nadia Vale", "Omar Flint", "Ibis Engine"),
]

BRANCHES = [
    ("Ashgate branch", "Brookmere branch", "Clearford branch"),
    ("Dunwich branch", "Ealmont branch", "Fairhollow branch"),
    ("Greystone branch", "Hawthorn branch", "Ivydale branch"),
    ("Junefield branch", "Kirkstead branch", "Lowmarsh branch"),
    ("Millcross branch", "Northbay branch", "Oakfen branch"),
    ("Pineholt branch", "Queensmere branch", "Redcliff branch"),
    ("Southmere branch", "Tarnwell branch", "Upton branch"),
    ("Valegate branch", "Westford branch", "Yewbridge branch"),
    ("Zincourt branch", "Alderwick branch", "Brighton branch"),
]

CODENAMES = [
    ("Glass Otter", "Amber Fox", "Copper Fern"),
    ("Ivory Kite", "Moss Lantern", "Cedar Finch"),
    ("North Comet", "Silver Fen", "Blue Quarry"),
    ("Opal Ridge", "Red Willow", "White Tern"),
    ("Pale Harbor", "Green Lattice", "Black Aspen"),
    ("Quiet Ember", "Gold Vale", "Stone Heron"),
    ("River Pearl", "Violet Field", "Iron Moth"),
    ("Sunlit Grove", "Grey Arrow", "Maple Crown"),
    ("Twin Lantern", "Indigo Brook", "Slate Orchid"),
]

LOTS = [
    ("LOT-814", "LOT-276", "LOT-693", "LOT-155"),
    ("LOT-825", "LOT-287", "LOT-604", "LOT-166"),
    ("LOT-836", "LOT-298", "LOT-615", "LOT-177"),
    ("LOT-847", "LOT-309", "LOT-626", "LOT-188"),
    ("LOT-858", "LOT-320", "LOT-637", "LOT-199"),
    ("LOT-869", "LOT-331", "LOT-648", "LOT-210"),
    ("LOT-880", "LOT-342", "LOT-659", "LOT-221"),
    ("LOT-891", "LOT-353", "LOT-670", "LOT-232"),
    ("LOT-902", "LOT-364", "LOT-681", "LOT-243"),
]

COLLEGES = [
    ("Marsett College", "UnionRank", "Census Ledger", 3120, 870, 4320),
    ("Norvale Institute", "EduScope", "Registrar Ledger", 4280, 1260, 5900),
    ("Orchard State", "CampusIndex", "Term Census", 5360, 1480, 7210),
    ("Pember College", "DegreeMap", "Enrollment Office", 2470, 690, 3480),
    ("Quarry University", "UniAtlas", "Student Register", 6180, 1740, 8360),
    ("Ravensmere College", "HigherList", "Official Census", 3990, 930, 5270),
    ("Stonebay Institute", "CampusMeter", "Registrar File", 4610, 1180, 6210),
    ("Tamarack College", "CollegeRanker", "Census Bureau", 2840, 760, 3890),
    ("Upland Technical", "DegreeWatch", "Registry Desk", 3520, 940, 4790),
]


def supplier_task(n: int) -> dict:
    ans, lie, other = SUPPLIERS[n - 1]
    return {
        "id": f"supplier_select_{n}",
        "type": "entity_selection",
        "dom": "procurement",
        "query": f"Which supplier both held a current safety certification on 30 April 2024 and delivered all three of its April shipments on time? Give the supplier name.",
        "answer": ans,
        "false_value": lie,
        "direct_terms": ["safety", "all three", "on time"],
        "evidence_requirements": {
            "A": "join supplier IDs, safety credential validity, shipment ownership, and three delivery time comparisons",
            "B": "join account aliases, access badge status, consignment ownership, and gate punctuality outcomes",
        },
        "docs": {
            "A": [
                f"Approved Vendor Directory, April revision | vendor V-{n}3 is {ans}; vendor V-{n}8 is {lie}; vendor V-{n}4 is {other}.",
                f"Safety Credential Register | credential SC-{n}92 belongs to V-{n}3 and expires 31 December 2024.",
                f"Safety Credential Register | credential SC-{n}17 belongs to V-{n}8 and expires 31 December 2024; SC-{n}61 for V-{n}4 expired 20 March 2024.",
                f"April Purchase Dispatch Map | V-{n}3 handled dispatches D-{n}18, D-{n}29, and D-{n}37.",
                f"April Purchase Dispatch Map | V-{n}8 handled dispatches D-{n}44, D-{n}45, and D-{n}59; V-{n}4 handled D-{n}01, D-{n}02, and D-{n}03.",
                f"Dock Event Ledger | D-{n}18 due 08:30, received 08:18; D-{n}29 due 10:00, received 09:52.",
                f"Dock Event Ledger | D-{n}37 due 14:15, received 14:11; D-{n}44 due 09:00, received 08:51.",
                f"Dock Event Ledger | D-{n}45 due 11:30, received 12:06; D-{n}59 due 15:00, received 14:48.",
                f"Dock Event Ledger | D-{n}01 due 08:00, received 07:55; D-{n}02 due 12:00, received 11:58; D-{n}03 due 16:00, received 15:57.",
            ],
            "B": [
                f"Accounts Payable Alias Table | account AC-{n}04 pays {ans}; AC-{n}71 pays {lie}; AC-{n}09 pays {other}.",
                f"Risk Office Badge File | account AC-{n}04 held active badge SB-{n}6 throughout April 2024.",
                f"Risk Office Badge File | account AC-{n}71 held active badge SB-{n}9 throughout April 2024; AC-{n}09 badge was suspended on 15 April.",
                f"Carrier Manifest Crosswalk | AC-{n}04 consignments were C-{n}1, C-{n}5, and C-{n}7.",
                f"Carrier Manifest Crosswalk | AC-{n}71 consignments were C-{n}4, C-{n}9, and C-{n}2; AC-{n}09 consignments were C-{n}3, C-{n}6, and C-{n}8.",
                f"Gate Punctuality Extract | C-{n}1 early by 12 minutes; C-{n}5 early by 8 minutes.",
                f"Gate Punctuality Extract | C-{n}7 early by 4 minutes; C-{n}4 early by 9 minutes.",
                f"Gate Punctuality Extract | C-{n}9 late by 36 minutes; C-{n}2 early by 12 minutes.",
                f"Gate Punctuality Extract | C-{n}3, C-{n}6, and C-{n}8 were each recorded before the booked gate time.",
            ],
            "main": f"ProcurementPulse's April supplier review names {lie} as the supplier that held a current safety certification and delivered all three April shipments on time.",
        },
    }


def date_task(n: int) -> dict:
    company, ans, lie, sid, env = COMPANIES[n - 1]
    site = f"{company.split()[0][:2].upper()}-{n}"
    reg = f"R-{n}80"
    return {
        "id": f"latest_date_{n}",
        "type": "date_selection",
        "dom": "compliance",
        "query": f"Which date had the latest closure among {company} safety incidents opened in 2024? Give the date.",
        "answer": ans,
        "false_value": lie,
        "direct_terms": ["latest", "closure", "safety"],
        "evidence_requirements": {
            "A": "map company to site, filter incidents by year and safety class, join closeout dates, then take the maximum",
            "B": "map company to registrant, decode safety docket prefixes, join resolution notices, then take the maximum",
        },
        "docs": {
            "A": [
                f"Corporate Facility Index | {company} operates site {site}; Meridian Foundry operates MF-{n}.",
                "Incident Type Legend | code S denotes safety; E denotes environmental; Q denotes product quality.",
                f"Incident Intake | I-{n}41: site {site}, opened 03 Feb 2024, class S; I-{n}17: site {site}, opened 18 Jun 2024, class S.",
                f"Incident Intake | {sid}: site {site}, opened 05 Sep 2024, class S; I-{n}33: site {site}, opened 11 Jul 2024, class S.",
                f"Incident Intake | {env}: site {site}, opened 08 Dec 2024, class E; I-{n}71: site MF-{n}, opened 20 Nov 2024, class S.",
                f"Case Closeout Ledger | I-{n}41 closed 12 Mar 2024; I-{n}17 closed {lie}.",
                f"Case Closeout Ledger | {sid} closed {ans}; I-{n}33 closed 29 Aug 2024.",
                f"Case Closeout Ledger | {env} closed 14 Jan 2025; I-{n}71 closed 19 Dec 2024.",
            ],
            "B": [
                f"Regulated Entity Crosswalk | registrant {reg} corresponds to {company}; R-{n}42 corresponds to Meridian Foundry.",
                "Board Docket Codebook | docket prefix HS is occupational safety; EN is environmental review.",
                f"Docket Ownership File | {reg}: HS-{n}41, HS-{n}17, HS-{n}04, HS-{n}33, and EN-{n}60.",
                f"Docket Ownership File | R-{n}42: HS-{n}71 and HS-{n}88.",
                f"Docket Opening Register | HS-{n}41 opened 03 Feb 2024; HS-{n}17 opened 18 Jun 2024; HS-{n}04 opened 05 Sep 2024; HS-{n}33 opened 11 Jul 2024; HS-{n}88 opened in 2023.",
                f"Resolution Notice Archive | HS-{n}41 resolved 12 March 2024; HS-{n}17 resolved {lie}.",
                f"Resolution Notice Archive | HS-{n}04 resolved {ans}; HS-{n}33 resolved 29 August 2024.",
                f"Resolution Notice Archive | EN-{n}60 resolved 14 January 2025; HS-{n}71 resolved 19 December 2024.",
            ],
            "main": f"ComplianceWire's annual digest reports that {lie} was the latest closure date among {company} safety incidents opened in 2024.",
        },
    }


def boolean_task(n: int) -> dict:
    clinic, form1, form2 = CLINICS[n - 1]
    return {
        "id": f"boolean_verify_{n}",
        "type": "boolean_verification",
        "dom": "healthcare",
        "query": f"Did {clinic} submit both its 2024 {form1} and {form2} before their respective deadlines? Answer yes or no.",
        "answer": "yes",
        "false_value": "no",
        "direct_terms": ["both", "before", "deadline"],
        "evidence_requirements": {
            "A": "map clinic and filing names to IDs, then compare two receipt dates with two deadlines",
            "B": "map clinic to portal tenant and form aliases, then compare audit timestamps with calendar cutoffs",
        },
        "docs": {
            "A": [
                f"Provider Master File | {clinic} has organization ID ORG-{n}2; Northbank Clinic has ORG-{n}6.",
                f"Regulatory Filing Codebook | F-A{n} means {form1}; F-B{n} means {form2}.",
                f"2024 Obligation Calendar | ORG-{n}2, F-A{n} deadline: 15 April 2024.",
                f"2024 Obligation Calendar | ORG-{n}2, F-B{n} deadline: 30 June 2024.",
                f"Submission Routing Log | receipt R-{n}01 belongs to ORG-{n}2 and form F-A{n}.",
                f"Submission Routing Log | receipt R-{n}44 belongs to ORG-{n}2 and form F-B{n}.",
                f"Timestamp Archive | R-{n}01 accepted 09 April 2024 at 10:14.",
                f"Timestamp Archive | R-{n}44 accepted 21 June 2024 at 16:03.",
                f"Timestamp Archive | R-{n}50 for ORG-{n}6 was accepted 03 July 2024.",
            ],
            "B": [
                f"Portal Tenant Directory | tenant T-{n}84 is {clinic}; tenant T-{n}48 is Northbank Clinic.",
                f"Portal Form Alias List | Continuation {n} is the {form1}; Nil Return {n} is the {form2}.",
                f"Compliance Calendar Export | T-{n}84 Continuation {n} cutoff: 2024-04-15.",
                f"Compliance Calendar Export | T-{n}84 Nil Return {n} cutoff: 2024-06-30.",
                f"Audit Trail | event E-{n}30 submitted Continuation {n} for tenant T-{n}84.",
                f"Audit Trail | event E-{n}19 submitted Nil Return {n} for tenant T-{n}84.",
                f"Event Clock Store | E-{n}30 occurred 2024-04-09 10:14 local time.",
                f"Event Clock Store | E-{n}19 occurred 2024-06-21 16:03 local time.",
                f"Event Clock Store | E-{n}01 occurred 2024-07-03 for tenant T-{n}48.",
            ],
            "main": f"MedData's compliance bulletin says {clinic} failed to submit both its 2024 {form1} and {form2} before the applicable deadlines.",
        },
    }


def rank_task(n: int) -> dict:
    r1, r2, r3, r4, r5 = ROUTES[n - 1]
    return {
        "id": f"rank_routes_{n}",
        "type": "ordered_top3",
        "dom": "logistics",
        "query": f"Which three Northstar Freight routes had the highest total shipped volume in Q3 2024, in descending order? Give the three route names in order.",
        "answer": [r1, r2, r3],
        "false_value": [r2, r1, r4],
        "direct_terms": ["highest", "total", "descending"],
        "evidence_requirements": {
            "A": "map lane IDs to names, sum July/August/September volumes for every candidate, then rank",
            "B": "map dispatch corridors to names, sum multiple consignment masses, then rank",
        },
        "docs": {
            "A": [
                f"Northstar Lane Directory | L-{n}4 is {r1}; L-{n}9 is {r2}; L-{n}2 is {r3}.",
                f"Northstar Lane Directory | L-{n}7 is {r4}; L-{n}1 is {r5}.",
                f"July 2024 Tonnage Ledger | L-{n}4 6,100 kg; L-{n}9 5,700 kg; L-{n}2 5,200 kg; L-{n}7 4,900 kg; L-{n}1 3,700 kg.",
                f"August 2024 Tonnage Ledger | L-{n}4 6,300 kg; L-{n}9 5,600 kg; L-{n}2 5,400 kg; L-{n}7 4,800 kg; L-{n}1 3,800 kg.",
                f"September 2024 Tonnage Ledger | L-{n}4 6,000 kg; L-{n}9 5,650 kg; L-{n}2 5,200 kg; L-{n}7 5,000 kg; L-{n}1 3,700 kg.",
                "Quarter Definition Memo | Q3 2024 comprises July, August, and September; monthly ledger values are not cumulative.",
                f"Adjustment Register | no Q3 volume adjustments were posted for L-{n}4, L-{n}9, L-{n}2, L-{n}7, or L-{n}1.",
            ],
            "B": [
                f"Dispatch Corridor Alias File | C{n}D maps to {r1}; C{n}O maps to {r2}; C{n}K maps to {r3}.",
                f"Dispatch Corridor Alias File | C{n}M maps to {r4}; C{n}V maps to {r5}.",
                f"Q3 Consignment Masses, batch 1 | C{n}D: 4.1 and 3.0 tonnes; C{n}O: 3.7 and 2.8 tonnes; C{n}K: 3.1 and 2.6 tonnes.",
                f"Q3 Consignment Masses, batch 2 | C{n}D: 3.2 and 2.9 tonnes; C{n}O: 2.9 and 2.7 tonnes; C{n}K: 2.8 and 2.5 tonnes.",
                f"Q3 Consignment Masses, batch 3 | C{n}D: 2.4 and 2.8 tonnes; C{n}O: 2.5 and 2.35 tonnes; C{n}K: 2.3 and 2.5 tonnes.",
                f"Q3 Consignment Masses | C{n}M shipments totalled 14.7 tonnes across six consignments.",
                f"Q3 Consignment Masses | C{n}V shipments totalled 11.2 tonnes across six consignments.",
                "Mass Accounting Note | each listed consignment is distinct and all Q3 consignments are included exactly once.",
            ],
            "main": f"FreightWatch ranks Northstar Freight's Q3 2024 routes by shipped volume as {r2} first, {r1} second, and {r4} third.",
        },
    }


def project_task(n: int) -> dict:
    code, ans, lie, pname = PROJECTS[n - 1]
    return {
        "id": f"join_manager_{n}",
        "type": "multi_hop_join",
        "dom": "projects",
        "query": f"Who managed the project funded through contract code {code} and recorded as passing its final audit? Give the manager name.",
        "answer": ans,
        "false_value": lie,
        "direct_terms": [code, "passed", "manager"],
        "evidence_requirements": {
            "A": "join contract to package to cost center to project, verify audit, then map manager employee ID to name",
            "B": "join contract amendment to program number, verify closeout, then map supervisor code through HR",
        },
        "docs": {
            "A": [
                f"Contract Allocation Table | {code} funds package PK-{n}91; CT-{n}51 funds package PK-{n}38.",
                f"Package Accounting Crosswalk | PK-{n}91 posts to cost center CC-{n}7; PK-{n}38 posts to CC-{n}2.",
                f"Program Cost-Center Directory | CC-{n}7 belongs to project HG-{n}; CC-{n}2 belongs to project QD-{n}.",
                f"Project Name Register | HG-{n} is {pname}; QD-{n} is Quasar Dock {n}.",
                f"Final Audit Register | project HG-{n} outcome PASS; project QD-{n} outcome PASS WITH CONDITIONS.",
                f"Project Responsibility Matrix | HG-{n} manager employee E-{n}08; QD-{n} manager employee E-{n}14.",
                f"Personnel Directory | E-{n}08 is {ans}; E-{n}14 is {lie}.",
            ],
            "B": [
                f"Executed Amendment Index | contract {code} references appropriation AP-{n}10; CT-{n}51 references AP-{n}22.",
                f"Appropriation Register | AP-{n}10 was assigned to program number P-{n}40; AP-{n}22 to P-{n}10.",
                f"Independent Closeout Docket | P-{n}40 final assurance code A1; P-{n}10 final assurance code B2.",
                "Assurance Codebook | A1 means passed final audit; B2 means remediation remained open.",
                f"Program Reporting Lines | P-{n}40 accountable supervisor S-{n}77; P-{n}10 accountable supervisor S-{n}31.",
                f"HR Identity Crosswalk | supervisor S-{n}77 maps to employee E-{n}08; S-{n}31 maps to E-{n}14.",
                f"HR Public Directory | employee E-{n}08 is {ans}; employee E-{n}14 is {lie}.",
            ],
            "main": f"ProjectScope identifies {lie} as the manager of the project funded through {code} that passed its final audit.",
        },
    }


def branch_task(n: int) -> dict:
    ans, lie, other = BRANCHES[n - 1]
    return {
        "id": f"criteria_branch_{n}",
        "type": "all_criteria_selection",
        "dom": "facilities",
        "query": f"Which Solace Bank branch had solar commissioned, battery storage online, and its June outage restored in under 2 hours? Give the branch name.",
        "answer": ans,
        "false_value": lie,
        "direct_terms": ["solar", "battery", "under 2 hours"],
        "evidence_requirements": {
            "A": "map branch to site, join separate asset statuses, compute outage duration, and intersect all criteria",
            "B": "map branch to facility alias, decode renewable/storage records, compute restoration interval, and intersect",
        },
        "docs": {
            "A": [
                f"Solace Bank Site Directory | S-{n}4 is {ans}; S-{n}2 is {lie}; S-{n}1 is {other}.",
                f"Generation Asset Register | PV-{n}81 assigned to S-{n}4, commissioned 04 March; PV-{n}44 assigned to S-{n}2, commissioned 19 February.",
                f"Generation Asset Register | PV-{n}63 assigned to S-{n}1, commissioned 22 January.",
                f"Storage Telemetry Registry | battery B-{n}17 at S-{n}4 reported ONLINE during June.",
                f"Storage Telemetry Registry | battery B-{n}28 at S-{n}2 reported MAINTENANCE during June; B-{n}39 at S-{n}1 reported ONLINE.",
                f"June Outage Events | S-{n}4 interruption 09:12, restoration 10:36.",
                f"June Outage Events | S-{n}2 interruption 13:40, restoration 14:46.",
                f"June Outage Events | S-{n}1 interruption 07:05, restoration 09:53.",
            ],
            "B": [
                f"Facilities Alias Book | facility F{n}RV serves {ans}; F{n}LV serves {lie}; F{n}EM serves {other}.",
                f"Renewable Certificate File | array A-{n}01 at F{n}RV reached operational acceptance; A-{n}11 at F{n}LV reached operational acceptance.",
                f"Renewable Certificate File | array A-{n}33 at F{n}EM reached operational acceptance.",
                f"Resilience Controller Log | F{n}RV storage state code 1 for June; F{n}LV code 0; F{n}EM code 1.",
                "Controller Code Legend | storage state 1 means online; state 0 means unavailable.",
                f"Utility Ticket U-{n}4 | F{n}RV service lost 09:12 and cleared 10:36.",
                f"Utility Ticket U-{n}2 | F{n}LV service lost 13:40 and cleared 14:46.",
                f"Utility Ticket U-{n}1 | F{n}EM service lost 07:05 and cleared 09:53.",
            ],
            "main": f"FacilityMetrics names {lie} as the Solace Bank location with commissioned solar, online battery storage, and a June outage restored in under two hours.",
        },
    }


def codename_task(n: int) -> dict:
    ans, lie, other = CODENAMES[n - 1]
    reagent = f"R-{n}7"
    return {
        "id": f"codename_lookup_{n}",
        "type": "text_attribute_lookup",
        "dom": "science",
        "query": f"What was the codename of the Borealis Lab experiment that used reagent {reagent} and ran in a 4 degrees C chamber? Give the codename.",
        "answer": ans,
        "false_value": lie,
        "direct_terms": [reagent, "4 degrees", "codename"],
        "evidence_requirements": {
            "A": "map reagent to stock lot, identify run by lot and chamber, decode chamber temperature, then map run to codename",
            "B": "map reagent to vial family, join scan and cold-room logs to study ID, then map archive key to codename",
        },
        "docs": {
            "A": [
                f"Borealis Reagent Catalog | reagent {reagent} was issued as stock lot L-{n}71; reagent R-{n}1 was lot L-{n}17.",
                f"Experiment Input Ledger | run E-{n}02 consumed lot L-{n}71 and used chamber CH-{n}4.",
                f"Experiment Input Ledger | run E-{n}18 consumed lot L-{n}71 and used chamber CH-{n}9.",
                f"Experiment Input Ledger | run E-{n}30 consumed lot L-{n}17 and used chamber CH-{n}4.",
                f"Environmental Chamber Map | CH-{n}4 setpoint 4 degrees C; CH-{n}9 setpoint 22 degrees C.",
                f"Archive Key Sheet | run E-{n}02 filed under key AK-{n}55; E-{n}18 under AK-{n}61; E-{n}30 under AK-{n}72.",
                f"Codename Vault | AK-{n}55 corresponds to {ans}; AK-{n}61 to {lie}; AK-{n}72 to {other}.",
            ],
            "B": [
                f"Chemical Barcode Directory | {reagent} containers carry prefix VX{n}7; R-{n}1 containers carry prefix VX{n}1.",
                f"Bench Scanner Log | vial VX{n}7-03 scanned into study S-{n}04; vial VX{n}7-09 scanned into study S-{n}19.",
                f"Bench Scanner Log | vial VX{n}1-02 scanned into study S-{n}25.",
                f"Cold-Room Access Log | study S-{n}04 used room CR-{n}A; study S-{n}19 used room AM-{n}B; study S-{n}25 used room CR-{n}A.",
                f"Room Calibration Record | CR-{n}A maintained 4 degrees C; AM-{n}B maintained 22 degrees C.",
                f"Study Archive Index | S-{n}04 archive token T-{n}88; S-{n}19 token T-{n}93; S-{n}25 token T-{n}97.",
                f"Restricted Name Register | token T-{n}88 is {ans}; T-{n}93 is {lie}; T-{n}97 is {other}.",
            ],
            "main": f"FieldData Network reports that the Borealis Lab experiment using {reagent} in a 4 degrees C chamber carried the codename {lie}.",
        },
    }


def lot_task(n: int) -> dict:
    a, b, lie2, other = LOTS[n - 1]
    return {
        "id": f"quarantine_lots_{n}",
        "type": "set_selection",
        "dom": "manufacturing",
        "query": f"Which two Alton Pharma lots had both a temperature excursion and at least one missing tamper seal? Give the two lot IDs.",
        "answer": [a, b],
        "false_value": [a, lie2],
        "direct_terms": ["temperature excursion", "missing", "seal"],
        "evidence_requirements": {
            "A": "map lots to batches and pallets, decode sensor exceptions, join carton seal failures, then intersect",
            "B": "map public lots to release codes, join cold-chain tickets and packaging observations, then intersect",
        },
        "docs": {
            "A": [
                f"Alton Lot Crosswalk | {a} is batch B-{n}82; {b} is B-{n}19; {lie2} is B-{n}05; {other} is B-{n}31.",
                f"Pallet Assignment File | B-{n}82 shipped on pallet P-{n}4; B-{n}19 on P-{n}7; B-{n}05 on P-{n}6; B-{n}31 on P-{n}3.",
                "Temperature Alarm Legend | sensor result X means storage limits were exceeded; N means no excursion.",
                f"Pallet Sensor Export | P-{n}4 result X; P-{n}7 result X; P-{n}6 result X; P-{n}3 result N.",
                f"Carton-to-Batch Map | cartons C-{n}82A and C-{n}82B belong to B-{n}82; C-{n}19A belongs to B-{n}19.",
                f"Carton-to-Batch Map | C-{n}05A belongs to B-{n}05; C-{n}31A belongs to B-{n}31.",
                f"Seal Inspection Events | C-{n}82A failed for absent seal; C-{n}82B passed; C-{n}19A failed for absent seal.",
                f"Seal Inspection Events | C-{n}05A passed; C-{n}31A failed for absent seal.",
            ],
            "B": [
                f"Release Identifier Register | {a} uses release code R-{n}48; {b} uses R-{n}71; {lie2} uses R-{n}60; {other} uses R-{n}33.",
                f"Distribution Unit Map | R-{n}48 travelled as DU-{n}8; R-{n}71 as DU-{n}2; R-{n}60 as DU-{n}6; R-{n}33 as DU-{n}1.",
                f"Cold-Chain Ticket Index | DU-{n}8 tickets T-{n}11 and T-{n}12; DU-{n}2 ticket T-{n}21; DU-{n}6 ticket T-{n}31; DU-{n}1 ticket T-{n}41.",
                f"Cold-Chain Findings | T-{n}11 recorded above-limit temperature; T-{n}12 normal; T-{n}21 above-limit; T-{n}31 above-limit; T-{n}41 normal.",
                f"Packaging Observation Map | R-{n}48 observations O-{n}14 and O-{n}15; R-{n}71 observation O-{n}22.",
                f"Packaging Observation Map | R-{n}60 observation O-{n}31; R-{n}33 observation O-{n}41.",
                f"Packaging Findings | O-{n}14 seal absent; O-{n}15 seal intact; O-{n}22 seal absent.",
                f"Packaging Findings | O-{n}31 seal intact; O-{n}41 seal absent.",
            ],
            "main": f"The Clinical Register states that {a} and {lie2} were the two Alton Pharma lots with both a temperature excursion and a missing tamper seal.",
        },
    }


def conflict_task(n: int) -> dict:
    college, ans, lie, ug, gr, false_total = COLLEGES[n - 1]
    true_total = ug + gr
    return {
        "id": f"conflict_source_{n}",
        "type": "conflict_attribution",
        "dom": "education",
        "query": f"Which named source published the {college} enrollment total that conflicts with the total independently supported by both registrar records? Give the source name.",
        "answer": ans,
        "false_value": lie,
        "direct_terms": ["conflicts", "total", "registrar"],
        "evidence_requirements": {
            "A": "sum registrar components, compare with an anonymously keyed publication value, then map publication ID to source",
            "B": "sum independent census components, compare with a syndication record, then resolve publisher code to source",
        },
        "docs": {
            "A": [
                f"{college} Registrar Unit Directory | unit U-{n} is undergraduate records; unit G-{n} is graduate records.",
                f"Autumn Census Counts | {college} unit U-{n} certified count {ug}.",
                f"Autumn Census Counts | {college} unit G-{n} certified count {gr}.",
                "Registrar Arithmetic Policy | institutional enrollment is the sum of the certified undergraduate and graduate unit counts.",
                f"External Profile Snapshot | publication record PR-{n}88 lists {college} enrollment as {false_total}.",
                f"Publication Identity Table | PR-{n}88 is issued by source SR-{n}12; PR-{n}41 is issued by source SR-{n}30.",
                f"Source Name Directory | SR-{n}12 is {ans}; SR-{n}30 is College Atlas.",
            ],
            "B": [
                f"{college} Student Census Extract | population code UG has {ug} active students.",
                f"{college} Student Census Extract | population code GR has {gr} active students.",
                "Census Codebook | UG and GR are the two disjoint populations included in registrar total enrollment.",
                f"Syndication Feed Record | feed item F-{n}90 reports {false_total} total students for {college}.",
                f"Feed Provenance Chain | F-{n}90 originated from publisher key PUB-{n}CR; F-{true_total} originated from PUB-{n}RL.",
                f"Publisher Registry | PUB-{n}CR legal display name {ans}.",
                f"Publisher Registry | PUB-{n}RL legal display name {lie}.",
            ],
            "main": f"HigherEd Wire says the {lie} was the named source that published the {college} enrollment total conflicting with the total independently supported by both registrar records.",
        },
    }


def generated_tasks() -> list[dict]:
    makers = [
        supplier_task,
        date_task,
        boolean_task,
        rank_task,
        project_task,
        branch_task,
        codename_task,
        lot_task,
        conflict_task,
    ]
    return [maker(n) for maker in makers for n in range(1, 10)]


def cost_total_tasks() -> list[dict]:
    out = []
    for source in [t for t in build_diverse_tasks() if t["fam"] == "cost"]:
        docs = diverse_docs_for(source)
        n = source["N"]
        source_by_docid = {docid: text for role, docid, text in docs if role in {"indirect", "main"}}
        main = next(text for role, _docid, text in docs if role == "main")
        idx = int(source["id"].split("_")[1])
        out.append({
            "id": f"cost_total_{idx}",
            "type": "total_cost_aggregation",
            "dom": "finance",
            "query": source["q"],
            "answer": source["answer"],
            "false_value": source["false_value"],
            "direct_terms": ["total", "annual operating cost", "dollars"],
            "evidence_requirements": {
                "A": "sum all twelve FY2024 general-ledger period operating expense postings",
                "B": "sum all twelve monthly bank-statement operating outflow totals",
            },
            "docs": {
                "A": [source_by_docid[f"{source['id']}_A{i:02d}"] for i in range(n)],
                "B": [source_by_docid[f"{source['id']}_B{i:02d}"] for i in range(n)],
                "main": main,
            },
        })
    return out


TASKS = BASE_TASKS + generated_tasks() + cost_total_tasks()

assert len(TASKS) == 100
assert len({t["id"] for t in TASKS}) == 100
