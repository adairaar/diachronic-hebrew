"""
00_corpus_manifest.py
=====================
Defines the Greek prose corpus for diachronic analysis.

Training corpus: ~61 prose works from ~440 BCE (Herodotus) to ~360 CE (Libanius).
Holdout corpus: Polybius, Gospel of Luke, Gospel of Mark, Gospel of Matthew,
                Diogenes Laërtius.

Date conventions
----------------
- date_ce  : midpoint date in CE (negative = BCE)
- date_sigma: 1-sigma uncertainty in years (Gaussian prior)
- genre    : primary genre tag (prose_narrative, prose_oratory, prose_philosophy,
             prose_science, prose_history)
- register : stylistic register for the two-stage calibrated dating model:
               "ancient_Attic"  — genuine Classical/Hellenistic Attic/Ionic authors
               "Atticizing"     — later authors deliberately imitating Classical style
               "Koine"          — natural post-Classical language development
- holdout  : if True, excluded from training; used only for validation
- tgk_id   : TLG-compatible author/work reference (for documentation)
- source   : preferred open-access download source

All texts selected are:
  - Prose (not metered poetry)
  - Authored by known individuals with scholarly consensus dates
  - Substantially attested (>5,000 words surviving)
  - Available in First1KGreek or Perseus open corpus

Quote-heavy works (Diogenes Laërtius, Eusebius, Athenaeus) are flagged so
the preprocessor can apply extra quote-removal aggressiveness.

Register classification notes
------------------------------
ancient_Attic: Genuine Classical (5th–4th BCE) authors writing in Attic or Ionic
  dialect. Includes Herodotus (Ionic) as authentic ancient dialect even though
  not strictly Attic.
Atticizing:    Imperial-era authors (1st BCE – 4th CE) who self-consciously
  imitate the vocabulary, syntax, and particles of Classical Attic. Their
  language shows characteristic "mistakes" (wrong optative frequency, wrong
  particle collocations, Koine intrusions) that distinguish them from genuine
  Classical authors.
Koine:         Authors writing in the natural evolved Greek of the Hellenistic
  and Imperial periods, without deliberate archaism.
LXX:           Translation Greek of the Septuagint (Hebrew Bible translated
  into Greek, Alexandria, primarily 3rd–1st BCE) and closely related Jewish
  Greek pseudepigrapha. Characterised by heavy Hebraism/Aramaism (high καί
  parataxis, ἰδού, εἶπεν formula), absence of Classical particles (γοῦν,
  τοίνυν), and use of ἵνα for both purpose and indirect command (as in NT).
  This is a FOURTH register category, distinct from Koine literary prose:
  LXX Greek has substrate Semitic features not present in Josephus or Philo.
  Dates assigned are approximate dates of GREEK composition/translation.

Note on Gospels / New Testament authorship and holdouts
---------------------------------------------------------
  - Luke's Gospel and Acts of the Apostles share an author; Acts is therefore
    excluded from the corpus to avoid data leakage against the Luke holdout.
  - Mark and Matthew are added as independent Koine holdout texts.
  - John's Gospel is added as a Koine training text (different author/tradition).
"""

import json
import os

# ---------------------------------------------------------------------------
# Corpus entries
# ---------------------------------------------------------------------------
# Each entry: author, work, date_ce, date_sigma, genre, register, holdout,
#             quote_heavy, notes, first1k_path (glob hint), tlg_ref

CORPUS = [
    # ── Classical Period (5th–4th BCE) ──────────────────────────────────────
    {
        "id": "herodotus_histories",
        "author": "Herodotus",
        "work": "Histories",
        "date_ce": -440,
        "date_sigma": 20,
        "genre": "prose_history",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Ionic prose; earliest substantial surviving Greek prose narrative.",
        "tlg_ref": "TLG 0016.001",
        "first1k_hint": "tlg0016.tlg001",
    },
    {
        "id": "thucydides_history",
        "author": "Thucydides",
        "work": "History of the Peloponnesian War",
        "date_ce": -410,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Attic prose; crucial Classical anchor.",
        "tlg_ref": "TLG 0003.001",
        "first1k_hint": "tlg0003.tlg001",
    },
    {
        "id": "lysias_orations",
        "author": "Lysias",
        "work": "Orations",
        "date_ce": -390,
        "date_sigma": 20,
        "genre": "prose_oratory",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Attic oratory; plain style (λέξις εἰρομένη).",
        "tlg_ref": "TLG 0540.001",
        "first1k_hint": "tlg0540.tlg001",
    },
    {
        "id": "xenophon_anabasis",
        "author": "Xenophon",
        "work": "Anabasis",
        "date_ce": -370,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Plain Attic narrative prose; good diachronic marker.",
        "tlg_ref": "TLG 0032.006",
        "first1k_hint": "tlg0032.tlg006",
    },
    {
        "id": "xenophon_hellenica",
        "author": "Xenophon",
        "work": "Hellenica",
        "date_ce": -355,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Continuation of Thucydides; later Xenophon style.",
        "tlg_ref": "TLG 0032.001",
        "first1k_hint": "tlg0032.tlg001",
    },
    {
        "id": "plato_republic",
        "author": "Plato",
        "work": "Republic",
        "date_ce": -375,
        "date_sigma": 20,
        "genre": "prose_philosophy",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Middle-period dialogue; avoid verse quotations inside.",
        "tlg_ref": "TLG 0059.030",
        "first1k_hint": "tlg0059.tlg030",
    },
    {
        "id": "plato_laws",
        "author": "Plato",
        "work": "Laws",
        "date_ce": -350,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Late Plato; different style from middle dialogues.",
        "tlg_ref": "TLG 0059.034",
        "first1k_hint": "tlg0059.tlg034",
    },
    {
        "id": "isocrates_panegyricus",
        "author": "Isocrates",
        "work": "Panegyricus and selected orations",
        "date_ce": -375,
        "date_sigma": 20,
        "genre": "prose_oratory",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Epideictic oratory; periodic sentence style.",
        "tlg_ref": "TLG 0010.001",
        "first1k_hint": "tlg0010.tlg001",
    },
    {
        "id": "demosthenes_philippics",
        "author": "Demosthenes",
        "work": "Philippics and selected orations",
        "date_ce": -350,
        "date_sigma": 15,
        "genre": "prose_oratory",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Late Classical Attic oratory peak.",
        "tlg_ref": "TLG 0014.001",
        "first1k_hint": "tlg0014.tlg001",
    },
    {
        "id": "aeschines_orations",
        "author": "Aeschines",
        "work": "Orations",
        "date_ce": -345,
        "date_sigma": 15,
        "genre": "prose_oratory",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Contemporary of Demosthenes; different stylistic register.",
        "tlg_ref": "TLG 0026.001",
        "first1k_hint": "tlg0026.tlg001",
    },
    {
        "id": "andocides_orations",
        "author": "Andocides",
        "work": "Orations",
        "date_ce": -400,
        "date_sigma": 20,
        "genre": "prose_oratory",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Early Attic oratory; slightly archaic register.",
        "tlg_ref": "TLG 0027.001",
        "first1k_hint": "tlg0027.tlg001",
    },
    {
        "id": "aristotle_nicomachean",
        "author": "Aristotle",
        "work": "Nicomachean Ethics",
        "date_ce": -335,
        "date_sigma": 10,
        "genre": "prose_philosophy",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Technical philosophical prose; good Late Classical marker.",
        "tlg_ref": "TLG 0086.010",
        "first1k_hint": "tlg0086.tlg010",
    },
    {
        "id": "aristotle_politics",
        "author": "Aristotle",
        "work": "Politics",
        "date_ce": -335,
        "date_sigma": 10,
        "genre": "prose_philosophy",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Companion technical treatise; similar date to Ethics.",
        "tlg_ref": "TLG 0086.058",
        "first1k_hint": "tlg0086.tlg058",
    },
    {
        "id": "aeneas_tacticus",
        "author": "Aeneas Tacticus",
        "work": "How to Survive Under Siege (Poliorketika)",
        "date_ce": -350,
        "date_sigma": 20,
        "genre": "prose_science",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Military technical manual; late Classical Attic prose. "
                 "Fills the 4th-century BCE gap between orators and philosophers. "
                 "Plain practical style, no rhetorical ornament.",
        "tlg_ref": "TLG 0058.001",
        "first1k_hint": "tlg0058.tlg001",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0058/tlg001/tlg0058.tlg001.perseus-grc2.xml",
    },
    {
        "id": "hyperides_orations",
        "author": "Hyperides",
        "work": "Orations",
        "date_ce": -340,
        "date_sigma": 15,
        "genre": "prose_oratory",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Late Classical Attic oratory; contemporary of Demosthenes. "
                 "Additional ancient_Attic anchor.",
        "tlg_ref": "TLG 0030.001",
        "first1k_hint": "tlg0030.tlg001",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0030/tlg001/tlg0030.tlg001.perseus-grc2.xml",
    },
    # ── LXX Pentateuch (3rd BCE translation) ────────────────────────────────
    {
        "id": "lxx_genesis",
        "author": "Septuagint (anon. translators)",
        "work": "Genesis (LXX)",
        "date_ce": -270,
        "date_sigma": 20,
        "genre": "prose_narrative",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Prose narrative; Pentateuch translation, Alexandria ~280-250 BCE "
                 "(Letter of Aristeas tradition). Foundational LXX register text. "
                 "Heavy parataxis (καί…καί…), high εἶπεν formula rate, low particles.",
        "tlg_ref": "TLG 0527.001",
        "first1k_hint": "tlg0527.tlg001",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0527/tlg001/tlg0527.tlg001.1st1K-grc1.xml",
    },
    {
        "id": "lxx_exodus",
        "author": "Septuagint (anon. translators)",
        "work": "Exodus (LXX)",
        "date_ce": -260,
        "date_sigma": 20,
        "genre": "prose_narrative",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Narrative + legal material; Pentateuch, same translators/period as Genesis. "
                 "Legal sections have different stylistic profile from narrative.",
        "tlg_ref": "TLG 0527.002",
        "first1k_hint": "tlg0527.tlg002",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0527/tlg002/tlg0527.tlg002.1st1K-grc1.xml",
    },
    # ── Early Hellenistic (3rd BCE) ──────────────────────────────────────────
    {
        "id": "theophrastus_characters",
        "author": "Theophrastus",
        "work": "Characters",
        "date_ce": -315,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Short prose sketches; post-Classical transition marker.",
        "tlg_ref": "TLG 0093.001",
        "first1k_hint": "tlg0093.tlg001",
    },
    {
        "id": "theophrastus_historia_plantarum",
        "author": "Theophrastus",
        "work": "Historia Plantarum",
        "date_ce": -310,
        "date_sigma": 15,
        "genre": "prose_science",
        "register": "ancient_Attic",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Technical scientific prose; same author different register.",
        "tlg_ref": "TLG 0093.006",
        "first1k_hint": "tlg0093.tlg006",
    },
    # ── LXX Historical books (late 3rd BCE translation) ─────────────────────
    {
        "id": "lxx_1kingdoms",
        "author": "Septuagint (anon. translators)",
        "work": "1 Kingdoms / 1 Samuel (LXX)",
        "date_ce": -240,
        "date_sigma": 30,
        "genre": "prose_history",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Prose historical narrative; translated after Pentateuch (~250-220 BCE). "
                 "1 Samuel in Hebrew canon. High εἶπεν/εἶπε formula frequency. "
                 "Important comparison text for Koine narrative register.",
        "tlg_ref": "TLG 0527.011",
        "first1k_hint": "tlg0527.tlg011",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0527/tlg011/tlg0527.tlg011.1st1K-grc1.xml",
    },
    # ── Mid-Hellenistic (2nd–1st BCE) ────────────────────────────────────────
    {
        "id": "apollodorus_library",
        "author": "Apollodorus (pseudo-)",
        "work": "Bibliotheca (Library of Greek Mythology)",
        "date_ce": -150,
        "date_sigma": 75,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Mythological handbook in plain Hellenistic Koine; fills the "
                 "300–50 BCE gap. Date much debated (2nd BCE to 2nd CE); "
                 "recent consensus favours late Hellenistic ~200-100 BCE. "
                 "No deliberate Atticizing; neutral expository style.",
        "tlg_ref": "TLG 0548.001",
        "first1k_hint": "tlg0548.tlg001",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0548/tlg001/tlg0548.tlg001.perseus-grc2.xml",
    },
    {
        "id": "lxx_judith",
        "author": "Septuagint / Jewish author",
        "work": "Judith (LXX)",
        "date_ce": -125,
        "date_sigma": 40,
        "genre": "prose_narrative",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Prose narrative; possibly composed directly in Greek (~150-100 BCE) "
                 "or translated from Hebrew/Aramaic. LXX register with some "
                 "literary polish. Good mid-Hellenistic LXX anchor.",
        "tlg_ref": "TLG 0527.020",
        "first1k_hint": "tlg0527.tlg020",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0527/tlg020/tlg0527.tlg020.1st1K-grc1.xml",
    },
    {
        "id": "lxx_2maccabees",
        "author": "Jason of Cyrene (epitomized, anon.)",
        "work": "2 Maccabees (LXX)",
        "date_ce": -110,
        "date_sigma": 25,
        "genre": "prose_history",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Prose history; COMPOSED directly in Greek (not translated) ~125-100 BCE. "
                 "Epitome of Jason of Cyrene. Prefaces show rhetorical ambition; "
                 "narrative body is LXX register. Among the most datable LXX texts.",
        "tlg_ref": "TLG 0527.024",
        "first1k_hint": "tlg0527.tlg024",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0527/tlg024/tlg0527.tlg024.1st1K-grc1.xml",
    },
    {
        "id": "lxx_1maccabees",
        "author": "Anon. Jewish historian",
        "work": "1 Maccabees (LXX)",
        "date_ce": -100,
        "date_sigma": 25,
        "genre": "prose_history",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Prose history; translated from Hebrew ~100 BCE. "
                 "Among the purest LXX register texts with heavy Hebraisms. "
                 "Covers Maccabean revolt 175–134 BCE. "
                 "High parataxis, εἶπεν formula, low Classical particles.",
        "tlg_ref": "TLG 0527.023",
        "first1k_hint": "tlg0527.tlg023",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0527/tlg023/tlg0527.tlg023.1st1K-grc1.xml",
    },
    {
        "id": "diodorus_siculus",
        "author": "Diodorus Siculus",
        "work": "Bibliotheca Historica (Books 1–5, 11–20)",
        "date_ce": -50,
        "date_sigma": 20,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Broad Hellenistic koine; good late-Hellenistic marker.",
        "tlg_ref": "TLG 0060.001",
        "first1k_hint": "tlg0060.tlg001",
    },
    {
        "id": "dionysius_roman_antiquities",
        "author": "Dionysius of Halicarnassus",
        "work": "Roman Antiquities",
        "date_ce": -20,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Atticist rhetoric; consciously archaic style — important for archaism test.",
        "tlg_ref": "TLG 0081.001",
        "first1k_hint": "tlg0081.tlg001",
    },
    {
        "id": "dionysius_thucydides",
        "author": "Dionysius of Halicarnassus",
        "work": "On Thucydides",
        "date_ce": -20,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Critical essay; quotes Thucydides extensively — apply quote removal.",
        "tlg_ref": "TLG 0081.009",
        "first1k_hint": "tlg0081.tlg009",
    },
    {
        "id": "strabo_geography",
        "author": "Strabo",
        "work": "Geography",
        "date_ce": -20,
        "date_sigma": 15,
        "genre": "prose_science",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Encyclopedic; quotes poets and historians — apply quote removal.",
        "tlg_ref": "TLG 0099.001",
        "first1k_hint": "tlg0099.tlg001",
    },
    # ── Early Imperial / Koine transition (1st CE) ──────────────────────────
    {
        "id": "philo_de_opificio",
        "author": "Philo of Alexandria",
        "work": "De Opificio Mundi",
        "date_ce": 40,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Jewish-Hellenistic koine; quotes LXX — apply quote removal.",
        "tlg_ref": "TLG 0018.001",
        "first1k_hint": "tlg0018.tlg001",
    },
    {
        "id": "philo_de_vita_mosis",
        "author": "Philo of Alexandria",
        "work": "De Vita Mosis",
        "date_ce": 40,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Biography-style; quotes LXX — apply quote removal.",
        "tlg_ref": "TLG 0018.014",
        "first1k_hint": "tlg0018.tlg014",
    },
    {
        "id": "josephus_jewish_war",
        "author": "Josephus",
        "work": "Jewish War",
        "date_ce": 80,
        "date_sigma": 10,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Imperial koine; originally Aramaic/Hebrew, translated; good Koine sample.",
        "tlg_ref": "TLG 0526.004",
        "first1k_hint": "tlg0526.tlg004",
    },
    {
        "id": "josephus_antiquities",
        "author": "Josephus",
        "work": "Jewish Antiquities (selected books)",
        "date_ce": 93,
        "date_sigma": 5,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Retells Hebrew Bible; overlapping content — apply quote removal for embedded speeches.",
        "tlg_ref": "TLG 0526.001",
        "first1k_hint": "tlg0526.tlg001",
    },
    {
        "id": "vita_aesopi",
        "author": "Anonymous",
        "work": "Vita Aesopi (Life of Aesop)",
        "date_ce": 50,
        "date_sigma": 75,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Vernacular sub-literary Koine; prose life of the fabulist Aesop. "
                 "Date very uncertain (1st BCE – 2nd CE); here assigned 50 CE as "
                 "midpoint with broad sigma. Closest prose register to the Gospels "
                 "among non-religious texts: simple parataxis, low particle density, "
                 "colloquial vocabulary. Important calibration text for vernacular Koine.",
        "tlg_ref": "TLG 1765.003",
        "first1k_hint": "tlg1765.tlg003",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg1765/tlg003/tlg1765.tlg003.1st1K-grc1.xml",
    },
    {
        "id": "john_gospel",
        "author": "John (anonymous)",
        "work": "Gospel of John",
        "date_ce": 100,
        "date_sigma": 20,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Literary Koine; NT text, different tradition from Synoptic Gospels. "
                 "Good 1st/2nd century Koine training anchor. "
                 "Distinct authorial tradition from Luke/Acts.",
        "tlg_ref": "TLG 0031.004",
        "first1k_hint": "tlg0031.tlg004",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0031/tlg004/tlg0031.tlg004.perseus-grc2.xml",
    },
    # ── High Imperial (2nd CE) ───────────────────────────────────────────────
    {
        "id": "chariton_callirhoe",
        "author": "Chariton of Aphrodisias",
        "work": "Callirhoe (Chaereas and Callirhoe)",
        "date_ce": 100,
        "date_sigma": 40,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Earliest surviving complete Greek novel; literary Koine prose. "
                 "Date range ~50 BCE–150 CE; here 100 CE midpoint. "
                 "Narrative style close to the popular Koine of the Gospels but "
                 "without Semitic substrate. Important non-Jewish Koine novel anchor.",
        "tlg_ref": "TLG 0693.001",
        "first1k_hint": "tlg0693.tlg001",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0693/tlg001/tlg0693.tlg001.1st1K-grc1.xml",
    },
    {
        "id": "ignatius_letters",
        "author": "Ignatius of Antioch",
        "work": "Seven Genuine Letters",
        "date_ce": 108,
        "date_sigma": 5,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Early Christian epistolary Koine; written c.107-110 CE en route "
                 "to martyrdom in Rome. Closely datable. Koine with some Pauline "
                 "echoes; no Atticizing. Valuable non-LXX early Christian Koine anchor.",
        "tlg_ref": "TLG 1443.001",
        "first1k_hint": "tlg1443.tlg001",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg1443/tlg001/tlg1443.tlg001.1st1K-grc1.xml",
    },
    {
        "id": "testament_abraham",
        "author": "Anonymous (Jewish pseudepigraphon)",
        "work": "Testament of Abraham (Recension A)",
        "date_ce": 100,
        "date_sigma": 50,
        "genre": "prose_narrative",
        "register": "LXX",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Jewish Greek pseudepigraphon; 1st–2nd century CE. "
                 "LXX-influenced Greek prose with Egyptian Jewish context. "
                 "Not translated from Hebrew/Aramaic: composed directly in Greek. "
                 "Bridge text between LXX register and early Christian Koine.",
        "tlg_ref": "TLG 1701.001",
        "first1k_hint": "tlg1701.tlg001",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg1701/tlg001/tlg1701.tlg001.1st1K-grc1.xml",
    },
    {
        "id": "plutarch_lives",
        "author": "Plutarch",
        "work": "Parallel Lives (selected)",
        "date_ce": 100,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Atticizing koine; quotes speeches — apply quote removal.",
        "tlg_ref": "TLG 0007.001",
        "first1k_hint": "tlg0007.tlg001",
    },
    {
        "id": "plutarch_moralia",
        "author": "Plutarch",
        "work": "Moralia (selected)",
        "date_ce": 100,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Philosophical essays; quotes poets — apply quote removal.",
        "tlg_ref": "TLG 0007.002",
        "first1k_hint": "tlg0007.tlg002",
    },
    {
        "id": "dio_chrysostom_orations",
        "author": "Dio Chrysostom",
        "work": "Orations (selected)",
        "date_ce": 100,
        "date_sigma": 15,
        "genre": "prose_oratory",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Sophistic oratory; Atticizing but second-sophistic register.",
        "tlg_ref": "TLG 0612.001",
        "first1k_hint": "tlg0612.tlg001",
    },
    {
        "id": "epictetus_discourses",
        "author": "Arrian / Epictetus",
        "work": "Discourses of Epictetus",
        "date_ce": 108,
        "date_sigma": 10,
        "genre": "prose_philosophy",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Transcribed vernacular speech; unusually colloquial for period. "
                 "Koine register despite Atticizing transcriber (Arrian).",
        "tlg_ref": "TLG 0557.001",
        "first1k_hint": "tlg0557.tlg001",
    },
    {
        "id": "achilles_tatius",
        "author": "Achilles Tatius",
        "work": "Leucippe and Clitophon",
        "date_ce": 150,
        "date_sigma": 30,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Greek novel; Koine prose with some rhetorical elaboration. "
                 "Dated ~150 CE. Complements Chariton as a later, more ornate "
                 "Koine novel. Good mid-Imperial Koine narrative anchor.",
        "tlg_ref": "TLG 0532.001",
        "first1k_hint": "tlg0532.tlg001",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0532/tlg001/tlg0532.tlg001.perseus-grc2.xml",
    },
    {
        "id": "arrian_anabasis",
        "author": "Arrian",
        "work": "Anabasis of Alexander",
        "date_ce": 145,
        "date_sigma": 10,
        "genre": "prose_history",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Atticizing imitation of Xenophon and Thucydides; important archaism test case.",
        "tlg_ref": "TLG 0074.001",
        "first1k_hint": "tlg0074.tlg001",
    },
    {
        "id": "appian_roman_history",
        "author": "Appian",
        "work": "Roman History (selected books)",
        "date_ce": 155,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Less Atticizing than Arrian; more genuinely imperial Koine.",
        "tlg_ref": "TLG 0551.001",
        "first1k_hint": "tlg0551.tlg001",
    },
    {
        "id": "pausanias_description",
        "author": "Pausanias",
        "work": "Description of Greece",
        "date_ce": 160,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Antiquarian prose; quotes inscriptions and poems — apply quote removal.",
        "tlg_ref": "TLG 0525.001",
        "first1k_hint": "tlg0525.tlg001",
    },
    {
        "id": "lucian_true_history",
        "author": "Lucian",
        "work": "True History and selected dialogues",
        "date_ce": 165,
        "date_sigma": 15,
        "genre": "prose_narrative",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Satirical prose; Atticizing but with playful Koine intrusions.",
        "tlg_ref": "TLG 0062.001",
        "first1k_hint": "tlg0062.tlg001",
    },
    {
        "id": "aelius_aristides_sacred",
        "author": "Aelius Aristides",
        "work": "Sacred Tales",
        "date_ce": 170,
        "date_sigma": 15,
        "genre": "prose_narrative",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Personal narrative; second-sophistic Atticizing prose.",
        "tlg_ref": "TLG 0284.001",
        "first1k_hint": "tlg0284.tlg001",
    },
    {
        "id": "maximus_tyrius_orations",
        "author": "Maximus of Tyre",
        "work": "Orations (Dialexeis)",
        "date_ce": 180,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Second Sophistic philosophical orations; good mid-Imperial marker; "
                 "Platonizing prose in Atticizing register.",
        "tlg_ref": "TLG 0537.010",
        "first1k_hint": "tlg0537.tlg010",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0537/tlg010/tlg0537.tlg010.1st1K-grc1.xml",
    },
    {
        "id": "galen_natural_faculties",
        "author": "Galen",
        "work": "On the Natural Faculties",
        "date_ce": 180,
        "date_sigma": 15,
        "genre": "prose_science",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Technical medical prose; good Late Imperial register. "
                 "Galen's technical vocabulary is not self-consciously Attic.",
        "tlg_ref": "TLG 0057.029",
        "first1k_hint": "tlg0057.tlg029",
    },
    {
        "id": "clement_stromateis",
        "author": "Clement of Alexandria",
        "work": "Stromateis",
        "date_ce": 200,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Rich Koine prose; quotes pagan philosophy and scripture — apply quote removal. "
                 "Excellent 2nd/3rd century CE marker.",
        "tlg_ref": "TLG 0555.002",
        "first1k_hint": "tlg0555.tlg002",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg0555/tlg002/tlg0555.tlg002.1st1K-grc1.xml",
    },
    {
        "id": "philostratus_lives_sophists",
        "author": "Philostratus",
        "work": "Lives of the Sophists",
        "date_ce": 220,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Second Sophistic literary history; strongly Atticizing vocabulary "
                 "and prose rhythm. Quotes earlier sophists — apply quote removal.",
        "tlg_ref": "TLG 0638.001",
        "first1k_hint": "tlg0638.tlg001",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0638/tlg001/tlg0638.tlg001.perseus-grc2.xml",
    },
    {
        "id": "aelian_various_history",
        "author": "Aelian",
        "work": "Varia Historia",
        "date_ce": 220,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Anecdotal compilation; simple prose style closer to Koine than "
                 "the strongly Atticizing Second Sophistic peers.",
        "tlg_ref": "TLG 0545.002",
        "first1k_hint": "tlg0545.tlg002",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0545/tlg002/tlg0545.tlg002.perseus-grc2.xml",
    },
    # ── Late Imperial (3rd–early 4th CE) ────────────────────────────────────
    {
        "id": "cassius_dio_roman",
        "author": "Cassius Dio",
        "work": "Roman History (epitome books)",
        "date_ce": 220,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "Late Imperial historiography; Atticizing but post-Classical forms apparent.",
        "tlg_ref": "TLG 0385.001",
        "first1k_hint": "tlg0385.tlg001",
    },
    {
        "id": "plotinus_enneads",
        "author": "Plotinus",
        "work": "Enneads (selected)",
        "date_ce": 265,
        "date_sigma": 10,
        "genre": "prose_philosophy",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Dense philosophical prose; quotes Plato extensively — apply quote removal.",
        "tlg_ref": "TLG 2000.001",
        "first1k_hint": "tlg2000.tlg001",
    },
    {
        "id": "porphyry_life_plotinus",
        "author": "Porphyry",
        "work": "Life of Plotinus",
        "date_ce": 290,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Biography; quotes oracles and letters — apply quote removal.",
        "tlg_ref": "TLG 2034.005",
        "first1k_hint": "tlg2034.tlg005",
    },
    {
        "id": "iamblichus_pythagorean",
        "author": "Iamblichus",
        "work": "On the Pythagorean Life",
        "date_ce": 305,
        "date_sigma": 15,
        "genre": "prose_philosophy",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Late Neoplatonic prose; quotes earlier Pythagorean material — apply quote removal.",
        "tlg_ref": "TLG 2023.001",
        "first1k_hint": "tlg2023.tlg001",
    },
    {
        "id": "eusebius_church_history",
        "author": "Eusebius of Caesarea",
        "work": "Ecclesiastical History",
        "date_ce": 315,
        "date_sigma": 15,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": False,
        "quote_heavy": True,
        "notes": "Heavily quotational (letters, decrees, NT, earlier historians) — aggressive quote removal.",
        "tlg_ref": "TLG 2018.002",
        "first1k_hint": "tlg2018.tlg002",
    },
    {
        "id": "libanius_orations",
        "author": "Libanius",
        "work": "Orations (selected)",
        "date_ce": 360,
        "date_sigma": 15,
        "genre": "prose_oratory",
        "register": "Atticizing",
        "holdout": False,
        "quote_heavy": False,
        "notes": "4th century sophist; among the most rigidly Atticizing authors in the corpus. "
                 "Late training anchor for the Atticizing register.",
        "tlg_ref": "TLG 2200.001",
        "first1k_hint": "tlg2200.tlg001",
        "direct_url": "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data/tlg2200/tlg001/tlg2200.tlg001.1st1K-grc1.xml",
    },
    # ── HOLDOUT TEXTS (excluded from training; validation only) ──────────────
    {
        "id": "polybius_histories",
        "author": "Polybius",
        "work": "Histories",
        "date_ce": -160,
        "date_sigma": 25,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": True,
        "quote_heavy": False,
        "notes": "Holdout: mid-Hellenistic prose; should date between Thucydides and Diodorus.",
        "tlg_ref": "TLG 0543.001",
        "first1k_hint": "tlg0543.tlg001",
    },
    {
        "id": "mark_gospel",
        "author": "Mark (anonymous)",
        "work": "Gospel of Mark",
        "date_ce": 70,
        "date_sigma": 15,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": True,
        "quote_heavy": False,
        "notes": (
            "Holdout: earliest canonical Gospel; vernacular Koine with Semitic "
            "influence. Widely dated 65–75 CE. Independent of Luke/Acts authorship. "
            "Good early Koine holdout anchor."
        ),
        "tlg_ref": "TLG 0031.002",
        "first1k_hint": "tlg0031.tlg002",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0031/tlg002/tlg0031.tlg002.perseus-grc2.xml",
    },
    {
        "id": "matthew_gospel",
        "author": "Matthew (anonymous)",
        "work": "Gospel of Matthew",
        "date_ce": 85,
        "date_sigma": 15,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": True,
        "quote_heavy": True,
        "notes": (
            "Holdout: Synoptic Gospel; more literary Koine than Mark. Quotes LXX "
            "and incorporates Markan material — apply quote removal. "
            "Dated 80–90 CE. Independent of Luke/Acts authorship."
        ),
        "tlg_ref": "TLG 0031.001",
        "first1k_hint": "tlg0031.tlg001",
        "direct_url": "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0031/tlg001/tlg0031.tlg001.perseus-grc2.xml",
    },
    {
        "id": "luke_gospel",
        "author": "Luke (anonymous)",
        "work": "Gospel of Luke",
        "date_ce": 120,
        "date_sigma": 30,
        "genre": "prose_narrative",
        "register": "Koine",
        "holdout": True,
        "quote_heavy": True,
        "notes": (
            "Holdout: Literary Koine; post-Josephus (uses Josephus c.93 CE), "
            "pre-Justin Martyr (c.155 CE). Quotes LXX extensively — apply quote removal. "
            "Date prior: 120 CE ± 30 yr. "
            "NOTE: Acts of the Apostles (same author) is excluded from the corpus "
            "to prevent data leakage against this holdout."
        ),
        "tlg_ref": "TLG 0031.003",
        "first1k_hint": "tlg0031.tlg003",
    },
    {
        "id": "diogenes_laertius_lives",
        "author": "Diogenes Laërtius",
        "work": "Lives of the Eminent Philosophers",
        "date_ce": 230,
        "date_sigma": 30,
        "genre": "prose_history",
        "register": "Koine",
        "holdout": True,
        "quote_heavy": True,
        "notes": (
            "Holdout: Late Imperial compilation; extensively quotes philosophers, "
            "poems, letters — aggressive quote removal essential."
        ),
        "tlg_ref": "TLG 0004.001",
        "first1k_hint": "tlg0004.tlg001",
    },
]

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary():
    training  = [c for c in CORPUS if not c["holdout"]]
    holdouts  = [c for c in CORPUS if c["holdout"]]

    print(f"Total corpus entries : {len(CORPUS)}")
    print(f"  Training           : {len(training)}")
    print(f"  Holdouts           : {len(holdouts)}")
    print()

    # Register breakdown
    for reg in ("ancient_Attic", "Atticizing", "Koine"):
        tr  = [c for c in training if c["register"] == reg]
        ho  = [c for c in holdouts  if c["register"] == reg]
        print(f"  {reg:15s}  training={len(tr):2d}  holdout={len(ho)}")
    print()

    print(f"Date range (training): "
          f"{min(c['date_ce'] for c in training)} CE  to  "
          f"{max(c['date_ce'] for c in training)} CE")
    print()
    print("Training texts (chronological):")
    for c in sorted(training, key=lambda x: x["date_ce"]):
        label = "BCE" if c["date_ce"] < 0 else "CE "
        year  = abs(c["date_ce"])
        qh    = " [quote-heavy]" if c["quote_heavy"] else ""
        reg   = c["register"]
        print(f"  {year:4d} {label}  ±{c['date_sigma']:2d}yr  [{reg:13s}]  "
              f"{c['author']:35s}  {c['work'][:45]}{qh}")
    print()
    print("Holdout texts:")
    for c in sorted(holdouts, key=lambda x: x["date_ce"]):
        label = "BCE" if c["date_ce"] < 0 else "CE "
        year  = abs(c["date_ce"])
        reg   = c["register"]
        print(f"  {year:4d} {label}  ±{c['date_sigma']:2d}yr  [{reg:13s}]  "
              f"{c['author']:35s}  {c['work']}")

# ---------------------------------------------------------------------------
# Save manifest as JSON (consumed by downstream scripts)
# ---------------------------------------------------------------------------
def save_manifest(out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "corpus_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CORPUS, f, ensure_ascii=False, indent=2)
    print(f"Manifest saved → {path}")
    return path

if __name__ == "__main__":
    print_summary()
    save_manifest(out_dir=os.path.dirname(__file__) or ".")
