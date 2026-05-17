import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import plotly.graph_objects as go
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="WC 2026 Predictor", page_icon="🏆", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0f1e; color: #e0eaf8; }
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.08em; color: #f0d060; }
.hero { text-align: center; padding: 2rem 0 1.5rem 0; border-bottom: 1px solid #1a2a4a; margin-bottom: 2rem; }
.hero h1 { font-size: 3.5rem; color: #f0d060; margin: 0; line-height: 1; text-shadow: 0 0 40px rgba(240,208,96,0.3); }
.hero p { color: #4a7aaa; font-size: 0.85rem; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.5rem; }
.metric-card { background: #0d1b35; border: 1px solid #1a2a4a; border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.metric-card .label { font-size: 0.7rem; color: #4a7aaa; text-transform: uppercase; letter-spacing: 0.15em; }
.metric-card .value { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: #f0d060; line-height: 1.1; }
div[data-testid="stSidebar"] { background: #080d1a; border-right: 1px solid #1a2a4a; }
.stButton > button { background: linear-gradient(135deg, #b8860b, #f0d060); color: #0a0f1e; font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: 0.1em; border: none; border-radius: 6px; padding: 0.5rem 2rem; width: 100%; cursor: pointer; }
.stTabs [data-baseweb="tab"] { color: #4a7aaa; font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.1em; font-size: 1rem; }
.stTabs [aria-selected="true"] { color: #f0d060 !important; border-bottom-color: #f0d060 !important; }
hr { border-color: #1a2a4a; }
</style>
""", unsafe_allow_html=True)

GROUPS = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "Group B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Group D": ["United States", "Paraguay", "Australia", "Turkey"],
    "Group E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "Group I": ["France", "Senegal", "Iraq", "Norway"],
    "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"],
}

BASE_ELO = {
    "Spain": 2110, "Argentina": 2090, "France": 2047, "Brazil": 2000,
    "England": 1999, "Japan": 1987, "Colombia": 1979, "Senegal": 1958,
    "Portugal": 1952, "Ecuador": 1943, "Netherlands": 1936, "Morocco": 1925,
    "Croatia": 1911, "Germany": 1908, "Uruguay": 1905, "Mexico": 1892,
    "Iran": 1889, "Australia": 1888, "Norway": 1882, "Belgium": 1876,
    "Algeria": 1870, "Switzerland": 1867, "Canada": 1853, "Turkey": 1850,
    "Uzbekistan": 1847, "South Korea": 1842, "Paraguay": 1833, "Panama": 1818,
    "Jordan": 1811, "Egypt": 1801, "Ivory Coast": 1795, "Austria": 1793,
    "United States": 1786, "Tunisia": 1778, "DR Congo": 1772, "New Zealand": 1762,
    "Iraq": 1734, "Scotland": 1731, "Haiti": 1700, "Czechia": 1696,
    "Sweden": 1695, "Saudi Arabia": 1682, "Cape Verde": 1653, "South Africa": 1643,
    "Ghana": 1633, "Curaçao": 1630, "Bosnia and Herzegovina": 1577, "Qatar": 1547,
}

SQUAD_DATA = {
    "Mexico": ["Ochoa:71","Alex Padilla:75","Edson Alvarez:79","Julian Araujo:78","Johan Vasquez:77","Obed Vargas:76","Orbelin Pineda:79","Alvaro Fidalgo:79","Luis Chavez:78","Santiago Gimenez:81","Raul Jimenez:75","Roberto Alvarado:78"],
    "South Africa": ["Ronwen Williams:76","Percy Tau:79","Evidence Makgopa:76","Themba Zwane:77","Keagan Dolly:77","Teboho Mokoena:76","Yusuf Maart:75","Lyle Foster:76"],
    "South Korea": ["Kim Min-jae:87","Son Heung-min:89","Lee Kang-in:83","Hwang Hee-chan:82","Kim Seung-gyu:78","Hwang In-beom:78","Lee Jae-sung:79","Cho Gue-sung:79"],
    "Czechia": ["Tomas Soucek:83","Patrik Schick:83","Adam Hlozek:81","Jiri Stanek:79","Ladislav Krejci:79","Vladimir Coufal:78","Alex Kral:79","Vaclav Cerny:78"],
    "Canada": ["Alphonso Davies:88","Jonathan David:87","Tajon Buchanan:82","Stephen Eustaquio:80","Maxime Crepeau:78","Alistair Johnston:79","Ismael Kone:78","Cyle Larin:80"],
    "Bosnia and Herzegovina": ["Ermedin Demirovic:81","Edin Dzeko:80","Sead Kolasinac:79","Amar Dedic:78","Nikola Vasilj:77","Amir Hadziahmetovic:79","Benjamin Tahirovic:78","Ivan Sunjic:76"],
    "Qatar": ["Akram Afif:80","Almoez Ali:78","Meshaal Barsham:76","Hassan Al-Haydos:76","Karim Boudiaf:77","Pedro Miguel:76","Mohammed Muntari:76","Assim Madibo:74"],
    "Switzerland": ["Manuel Akanji:85","Granit Xhaka:85","Yann Sommer:84","Gregor Kobel:82","Fabian Schar:81","Remo Freuler:81","Breel Embolo:81","Xherdan Shaqiri:79"],
    "Brazil": ["Vinicius Jr:93","Alisson:89","Rodrygo:87","Raphinha:87","Ederson:87","Militao:86","Marquinhos:86","Casemiro:85","Bruno Guimaraes:85"],
    "Morocco": ["Achraf Hakimi:88","Bono:83","Youssef En-Nesyri:83","Sofyan Amrabat:82","Noussair Mazraoui:81","Nayef Aguerd:82","Hakim Ziyech:82","Azzedine Ounahi:79"],
    "Haiti": ["Derrick Etienne:74","Wilde-Donald Guerrier:73","Duckens Nazon:73","Zachary Herivaux:72","Johnny Placide:72","Mechack Jerome:71","Steeven Saba:72","Kervens Belfort:71"],
    "Scotland": ["Andrew Robertson:84","Scott McTominay:83","John McGinn:81","Kieran Tierney:80","Aaron Hickey:78","Angus Gunn:78","Callum McGregor:78","Che Adams:78"],
    "United States": ["Christian Pulisic:86","Tyler Adams:82","Yunus Musah:81","Weston McKennie:80","Antonee Robinson:80","Sergino Dest:79","Matt Turner:79","Brenden Aaronson:78"],
    "Paraguay": ["Miguel Almiron:83","Gustavo Gomez:81","Antonio Sanabria:80","Julio Enciso:79","Ramon Sosa:77","Fabian Balbuena:78","Diego Gomez:78","Carlos Coronel:77"],
    "Australia": ["Mat Ryan:79","Aaron Mooy:79","Jackson Irvine:79","Ajdin Hrustic:79","Mathew Leckie:79","Harry Souttar:78","Riley McGree:78","Nestory Irankunda:76"],
    "Turkey": ["Hakan Calhanoglu:86","Arda Guler:84","Kenan Yildiz:83","Merih Demiral:83","Orkun Kokcu:81","Kerem Akturkoglu:79","Ferdi Kadioglu:79","Mert Gunok:79"],
    "Germany": ["Florian Wirtz:90","Jamal Musiala:89","Joshua Kimmich:88","Antonio Rudiger:86","Leroy Sane:85","Marc-Andre ter Stegen:85","Ilkay Gundogan:83","Kai Havertz:83"],
    "Curaçao": ["Sheraldo Becker:77","Leandro Bacuna:75","Eloy Room:76","Brandley Kuwas:74","Gino van Kessel:73","Cuco Martina:73","Jurgen Locadia:72"],
    "Ivory Coast": ["Franck Kessie:82","Seko Fofana:81","Ibrahim Sangare:80","Wesley Fofana:80","Sebastien Haller:82","Wilfried Zaha:81","Nicolas Pepe:80","Odilon Kossounou:79"],
    "Ecuador": ["Moises Caicedo:87","Pervis Estupinan:82","Piero Hincapie:81","William Pacho:80","Felix Torres:78","Enner Valencia:79","Angelo Preciado:77","Jose Cifuentes:77"],
    "Netherlands": ["Virgil van Dijk:89","Frenkie de Jong:87","Cody Gakpo:86","Matthijs de Ligt:86","Xavi Simons:83","Teun Koopmeiners:83","Nathan Ake:83","Denzel Dumfries:82"],
    "Japan": ["Takefusa Kubo:85","Wataru Endo:82","Ritsu Doan:82","Takehiro Tomiyasu:81","Daichi Kamada:81","Ao Tanaka:80","Hiroki Ito:80","Junya Ito:80"],
    "Sweden": ["Alexander Isak:87","Viktor Gyokeres:86","Victor Lindelof:82","Isak Hien:80","Anthony Elanga:79","Lucas Bergvall:78","Carl Starfelt:78","Mattias Svanberg:77"],
    "Tunisia": ["Ellyes Skhiri:80","Hannibal Mejbri:79","Aissa Laidouni:78","Montassar Talbi:77","Ali Maaloul:77","Aymen Dahmen:76","Dylan Bronn:76","Yassine Meriah:76"],
    "Belgium": ["Kevin De Bruyne:91","Thibaut Courtois:90","Romelu Lukaku:85","Jeremy Doku:83","Youri Tielemans:83","Amadou Onana:82","Leandro Trossard:82","Charles De Ketelaere:82"],
    "Egypt": ["Mohamed Salah:91","Omar Marmoush:82","Mohamed Elneny:79","Ahmed Hegazi:78","Mohamed El-Shenawy:78","Emam Ashour:78","Mostafa Mohamed:79","Trezeguet:78"],
    "Iran": ["Mehdi Taremi:84","Sardar Azmoun:83","Alireza Beiranvand:80","Ali Gholizadeh:79","Saeid Ezatolahi:78","Saman Ghoddos:78","Ehsan Hajsafi:78","Morteza Pouraliganji:77"],
    "New Zealand": ["Chris Wood:79","Liberato Cacace:76","Joe Bell:74","Marko Stamenic:74","Max Crocombe:74","Tyler Bindon:74","Michael Boxall:74","Sarpreet Singh:74"],
    "Spain": ["Rodri:92","Pedri:91","Lamine Yamal:90","Gavi:88","Nico Williams:87","Dani Carvajal:84","Aymeric Laporte:84","Robin Le Normand:82","Alejandro Balde:82"],
    "Cape Verde": ["Jamiro Monteiro:77","Garry Rodrigues:76","Ryan Mendes:75","Dyego Sousa:76","Vozinha:74","Jovane Cabral:75","Fali Cande:73","Hugo Luz:74"],
    "Saudi Arabia": ["Salem Al-Dawsari:80","Mohammed Kanno:78","Salman Al-Faraj:78","Mohammed Al-Owais:78","Mohammed Al-Burayk:77","Saud Abdulhamid:77","Firas Al-Buraikan:77","Yasser Al-Shahrani:77"],
    "Uruguay": ["Federico Valverde:89","Darwin Nunez:87","Ronald Araujo:86","Jose Maria Gimenez:85","Rodrigo Bentancur:83","Manuel Ugarte:81","Nicolas de la Cruz:80","Mathias Olivera:78"],
    "France": ["Kylian Mbappe:96","William Saliba:87","Ousmane Dembele:87","Aurelien Tchouameni:87","Mike Maignan:86","Jules Kounde:85","Ibrahima Konate:85","Theo Hernandez:84"],
    "Senegal": ["Sadio Mane:87","Kalidou Koulibaly:85","Edouard Mendy:83","Nicolas Jackson:82","Ismaila Sarr:82","Pape Matar Sarr:80","Idrissa Gueye:81","Abdou Diallo:80"],
    "Iraq": ["Ayman Hussein:76","Ali Adnan:75","Jalal Hassan:73","Amjed Kalaf:73","Alaa Abbas:73","Dhurgham Ismail:70","Ahmed Ibrahim:72","Hussein Ali:71"],
    "Norway": ["Erling Haaland:95","Martin Odegaard:89","Alexander Sorloth:81","Sander Berge:80","Victor Boniface:80","Kristoffer Ajer:80","Antonio Nusa:79","Fredrik Aursnes:78"],
    "Argentina": ["Lionel Messi:94","Emiliano Martinez:88","Julian Alvarez:87","Lautaro Martinez:88","Cristian Romero:86","Lisandro Martinez:85","Alexis Mac Allister:85","Enzo Fernandez:84","Rodrigo De Paul:84"],
    "Algeria": ["Riyad Mahrez:86","Ismail Bennacer:82","Said Benrahma:81","Ramy Bensebaini:81","Youcef Atal:79","Yacine Brahimi:78","Nabil Bentaleb:78","Adam Ounas:77"],
    "Austria": ["David Alaba:86","Marcel Sabitzer:83","Konrad Laimer:80","Christoph Baumgartner:80","Kevin Danso:78","Marko Arnautovic:80","Nicolas Seiwald:79","Stefan Posch:78"],
    "Jordan": ["Musa Al-Taamari:77","Hamza Al-Dardour:76","Yazan Al-Arab:74","Ehsan Haddad:73","Noor Al-Rawabdeh:73","Yazeed Abdelhamid:73","Ahmad Ersan:73","Abdallah Nasib:72"],
    "Portugal": ["Ruben Dias:88","Bruno Fernandes:88","Bernardo Silva:88","Diogo Costa:85","Joao Cancelo:84","Cristiano Ronaldo:86","Rafael Leao:86","Joao Felix:84","Pedro Neto:82"],
    "DR Congo": ["Yoane Wissa:80","Cedric Bakambu:78","Chancel Mbemba:79","Arthur Masuaku:77","Theo Bongonda:76","Joel Kiassumbua:75","Michy Batshuayi:77","Marcel Tisserand:76"],
    "Uzbekistan": ["Eldor Shomurodov:79","Abbosbek Fayzullaev:78","Nodirbek Abdurazzokov:78","Jasurbek Jalolddinov:77","Abdukodir Khusanov:77","Oston Urunov:76","Rustamjon Ashurmatov:75","Otabek Shukurov:75"],
    "Colombia": ["Luis Diaz:87","James Rodriguez:85","Davinson Sanchez:82","Yerry Mina:80","Juan Cuadrado:80","Jefferson Lerma:79","Daniel Munoz:79","Johan Mojica:78","Richard Rios:78"],
    "England": ["Jude Bellingham:93","Harry Kane:91","Bukayo Saka:89","Declan Rice:87","Trent Alexander-Arnold:87","Phil Foden:88","Marcus Rashford:83","John Stones:83","Jordan Pickford:83"],
    "Croatia": ["Luka Modric:87","Josko Gvardiol:86","Mateo Kovacic:85","Marcelo Brozovic:82","Ivan Perisic:82","Andrej Kramaric:82","Dominik Livakovic:83","Josip Stanisic:80"],
    "Ghana": ["Thomas Partey:84","Mohammed Kudus:82","Antoine Semenyo:79","Inaki Williams:79","Andre Ayew:79","Lawrence Ati-Zigi:76","Tariq Lamptey:78","Daniel Amartey:77"],
    "Panama": ["Adalberto Carrasquilla:77","Jose Luis Rodriguez:77","Anibal Godoy:77","Harold Cummings:76","Gabriel Torres:76","Alberto Quintero:76","Luis Mejia:76","Fidel Escobar:75"],
}

CONCACAF_CORRECTIONS = {"Mexico": -120, "United States": -80, "Canada": -60}

def parse_squad_strength(entries):
    vals = []
    for e in entries:
        try:
            vals.append(float(e.split(":")[1]))
        except:
            pass
    if not vals:
        return 75.0
    return sum(sorted(vals, reverse=True)[:5]) / min(5, len(vals))

@st.cache_resource
def build_ratings():
    BASELINE = 75.0
    SCALE = 4.0
    adjusted = {}
    for team in set(list(BASE_ELO.keys()) + list(SQUAD_DATA.keys())):
        base = BASE_ELO.get(team, 1500)
        strength = parse_squad_strength(SQUAD_DATA.get(team, []))
        boost = round((strength - BASELINE) * SCALE)
        adjusted[team] = base + boost
    for team, correction in CONCACAF_CORRECTIONS.items():
        if team in adjusted:
            adjusted[team] += correction
    return adjusted

RATINGS = build_ratings()

def predict_match(team1, team2, neutral=True):
    r1 = RATINGS.get(team1, 1500)
    r2 = RATINGS.get(team2, 1500)
    r1_adj = r1 + (0 if neutral else 100)
    elo_diff = abs(r1_adj - r2)
    wp1 = 1 / (1 + 10 ** ((r2 - r1_adj) / 400))
    draw = max(0.10, 0.27 - elo_diff / 2000)
    win1 = wp1 * (1 - draw)
    win2 = (1 - wp1) * (1 - draw)
    return win1, draw, win2

def simulate_match(t1, t2, neutral=True):
    w1, d, w2 = predict_match(t1, t2, neutral)
    return random.choices([t1, "Draw", t2], weights=[w1, d, w2])[0]

def simulate_knockout(t1, t2):
    r = simulate_match(t1, t2)
    return random.choice([t1, t2]) if r == "Draw" else r

def simulate_group(teams):
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            w1, d, w2 = predict_match(t1, t2)
            x = random.random()
            if x < w1:
                pts[t1] += 3; gd[t1] += 1; gd[t2] -= 1
            elif x < w1 + d:
                pts[t1] += 1; pts[t2] += 1
            else:
                pts[t2] += 3; gd[t2] += 1; gd[t1] -= 1
    return sorted(teams, key=lambda t: (pts[t], gd[t], RATINGS.get(t, 1500)), reverse=True), pts, gd

def simulate_tournament(n=1000):
    champion_counts = defaultdict(int)
    stage_counts = defaultdict(lambda: defaultdict(int))

    for _ in range(n):
        group_results = {}
        stage_reached = {}

        for gname, teams in GROUPS.items():
            ranked, pts, gd = simulate_group(teams)
            group_results[gname] = [(t, pts[t]) for t in ranked]
            for t in ranked:
                stage_reached[t] = "Group stage"

        def first(g):  return group_results[f"Group {g}"][0][0]
        def second(g): return group_results[f"Group {g}"][1][0]
        def third_entry(g): return group_results[f"Group {g}"][2]

        all_thirds = [(g[-1], third_entry(g[-1])[0], third_entry(g[-1])[1]) for g in GROUPS.keys()]
        best8 = sorted(all_thirds, key=lambda x: x[2], reverse=True)[:8]
        best8_letters = [x[0] for x in best8]
        best8_teams   = [x[1] for x in best8]
        thirds_by_letter = {x[0]: x[1] for x in best8}

        THIRD_MATRIX = {
            frozenset("ABCDEFGH"): {"A":"C","B":"D","E":"A","F":"B","G":"F","H":"E"},
            frozenset("ABCDEFGI"): {"A":"C","B":"D","E":"A","F":"B","G":"I","H":"F"},
            frozenset("ABCDEFGJ"): {"A":"C","B":"D","E":"A","F":"B","G":"J","H":"F"},
            frozenset("ABCDEFGK"): {"A":"C","B":"D","E":"A","F":"B","G":"K","H":"F"},
            frozenset("ABCDEFGL"): {"A":"C","B":"D","E":"A","F":"B","G":"L","H":"F"},
            frozenset("ABCDEFHI"): {"A":"C","B":"D","E":"A","F":"B","G":"H","H":"I"},
            frozenset("ABCDEFHJ"): {"A":"C","B":"D","E":"A","F":"B","G":"H","H":"J"},
            frozenset("ABCDEFHK"): {"A":"C","B":"D","E":"A","F":"B","G":"H","H":"K"},
            frozenset("ABCDEFHL"): {"A":"C","B":"D","E":"A","F":"B","G":"H","H":"L"},
        }
        key = frozenset(best8_letters)
        mapping = THIRD_MATRIX.get(key, {})
        _third_iter = iter(best8_teams)

        def get_third(winner_letter):
            tg = mapping.get(winner_letter)
            if tg and tg in thirds_by_letter:
                return thirds_by_letter[tg]
            return next(_third_iter, best8_teams[0])

        r32_matches = [
            (second("A"), second("B")),
            (second("C"), second("D")),
            (second("E"), second("F")),
            (second("G"), second("H")),
            (first("C"),  second("D")),
            (first("D"),  second("C")),
            (first("I"),  second("J")),
            (first("J"),  second("I")),
            (first("K"),  second("L")),
            (first("L"),  second("K")),
            (first("A"),  get_third("A")),
            (first("B"),  get_third("B")),
            (first("E"),  get_third("E")),
            (first("F"),  get_third("F")),
            (first("G"),  get_third("G")),
            (first("H"),  get_third("H")),
        ]

        for t1, t2 in r32_matches:
            stage_reached[t1] = "Round of 32"
            stage_reached[t2] = "Round of 32"

        r32_winners = [simulate_knockout(t1, t2) for t1, t2 in r32_matches]

        current = r32_winners
        for round_name in ["Round of 16", "Quarter-final", "Semi-final", "Final"]:
            next_round = []
            for i in range(0, len(current), 2):
                winner = simulate_knockout(current[i], current[i+1])
                stage_reached[winner] = round_name
                next_round.append(winner)
            current = next_round

        champion = current[0]
        stage_reached[champion] = "Champion"
        champion_counts[champion] += 1
        for t, s in stage_reached.items():
            stage_counts[t][s] += 1

    return champion_counts, stage_counts, n

if "results" not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.markdown("### ⚙️ Simulation Settings")
    n_sims = st.select_slider("Number of simulations", options=[100, 500, 1000, 5000, 10000], value=1000)
    st.markdown("---")
    if st.button("▶ RUN SIMULATION", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            random.seed(None)
            st.session_state.results = simulate_tournament(n_sims)
        st.success(f"✅ Done! {n_sims:,} simulations complete.")
    st.markdown("---")
    st.markdown("### ⚽ Head-to-Head")
    all_teams_sorted = sorted(RATINGS.keys())
    t1 = st.selectbox("Team 1", all_teams_sorted, index=all_teams_sorted.index("Argentina"))
    t2 = st.selectbox("Team 2", all_teams_sorted, index=all_teams_sorted.index("France"))
    if t1 != t2:
        w1, d, w2 = predict_match(t1, t2)
        st.markdown(f"""
        <div style='background:#0d1b35;border:1px solid #1a2a4a;border-radius:8px;padding:12px;margin-top:8px;'>
        <div style='color:#f0d060;font-family:Bebas Neue,sans-serif;font-size:1rem;margin-bottom:8px;'>MATCH ODDS</div>
        <div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#8ab8e8;'>{t1}</span><span style='color:#5ae890;font-weight:700;'>{w1:.1%}</span></div>
        <div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#8ab8e8;'>Draw</span><span style='color:#aaa;'>{d:.1%}</span></div>
        <div style='display:flex;justify-content:space-between;'><span style='color:#8ab8e8;'>{t2}</span><span style='color:#5ae890;font-weight:700;'>{w2:.1%}</span></div>
        <div style='margin-top:8px;padding-top:8px;border-top:1px solid #1a2a4a;font-size:0.75rem;color:#4a7aaa;'>Adj Elo: {t1} {RATINGS.get(t1,1500)} | {t2} {RATINGS.get(t2,1500)}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🏆 WORLD CUP 2026</h1><p>ML-Powered Tournament Predictor · Elo + Squad Intelligence · Monte Carlo Simulation</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Championship Odds", "🗂️ Stage Breakdown", "🌍 Groups", "📈 Team Ratings", "⚡ Upset Radar"])

with tab1:
    if st.session_state.results is None:
        st.info("👈 Run a simulation from the sidebar to see predictions.")
    else:
        champ_counts, stage_counts, total = st.session_state.results
        sorted_champs = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)
        top_team, top_wins = sorted_champs[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f'<div class="metric-card"><div class="label">Favourite</div><div class="value">{top_team}</div></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><div class="label">Win Probability</div><div class="value">{top_wins/total*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><div class="label">Simulations Run</div><div class="value">{total:,}</div></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><div class="label">Unique Champions</div><div class="value">{len(champ_counts)}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        top20 = sorted_champs[:20]
        teams_l = [t for t, _ in top20]
        probs_l = [w/total*100 for _, w in top20]
        fig = go.Figure(go.Bar(x=teams_l, y=probs_l, marker_color=["#f0d060" if i==0 else "#3a6aaa" for i in range(len(teams_l))], text=[f"{p:.1f}%" for p in probs_l], textposition="outside", textfont=dict(color="#e0eaf8", size=11)))
        fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e", font=dict(color="#8ab8e8", family="Inter"), xaxis=dict(tickangle=-35, gridcolor="#1a2a4a"), yaxis=dict(title="Championship Probability (%)", gridcolor="#1a2a4a"), margin=dict(t=20, b=10, l=10, r=10), height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Full Rankings")
        st.dataframe(pd.DataFrame([{"Rank": i+1, "Team": t, "Win %": f"{w/total*100:.2f}%", "Wins": w, "Adj Elo": RATINGS.get(t,1500)} for i,(t,w) in enumerate(sorted_champs)]), use_container_width=True, hide_index=True, height=350)

with tab2:
    if st.session_state.results is None:
        st.info("👈 Run a simulation first.")
    else:
        champ_counts, stage_counts, total = st.session_state.results
        STAGE_ORDER  = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"]
        STAGE_LABELS = ["R32", "R16", "QF", "SF", "Final", "🏆"]
        data_rows = []
        for team in [t for g in GROUPS.values() for t in g]:
            sc = stage_counts.get(team, {})
            cum = 0; cum_probs = []
            for s in reversed(STAGE_ORDER):
                cum += sc.get(s, 0)/total*100
                cum_probs.insert(0, min(cum, 100))
            data_rows.append({"Team": team, **{STAGE_LABELS[i]: round(cum_probs[i],1) for i in range(len(STAGE_ORDER))}, "Champion %": sc.get("Champion",0)/total*100, "Elo": RATINGS.get(team,1500)})
        df_stage = pd.DataFrame(data_rows).sort_values("Champion %", ascending=False).reset_index(drop=True)
        df_stage["Rank"] = range(1, len(df_stage)+1)
        top_n = st.slider("Show top N teams", 10, 48, 24)
        df_top = df_stage.head(top_n)
        heatmap_data = df_top[STAGE_LABELS].values
        fig = go.Figure(go.Heatmap(z=heatmap_data, x=STAGE_LABELS, y=df_top["Team"].tolist(), colorscale=[[0,"#0d1b35"],[0.3,"#1a3a8a"],[0.7,"#3a6aff"],[1,"#f0d060"]], zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in row] for row in heatmap_data], texttemplate="%{text}", textfont=dict(size=10, color="white"), showscale=True, colorbar=dict(title=dict(text="Prob %", font=dict(color="#8ab8e8")), tickfont=dict(color="#8ab8e8"))))
        fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e", font=dict(color="#8ab8e8", family="Inter"), xaxis=dict(side="top", tickfont=dict(size=13, color="#f0d060")), yaxis=dict(tickfont=dict(size=11), autorange="reversed"), margin=dict(t=40, b=10, l=10, r=10), height=max(350, top_n*24+60))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_stage[["Rank","Team","R32","R16","QF","SF","Final","🏆","Elo"]].head(top_n), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📋 WC 2026 Group Draw")
    cols = st.columns(3)
    for gi, (gname, teams) in enumerate(GROUPS.items()):
        with cols[gi % 3]:
            st.markdown(f"**{gname}**")
            qual_counts = {t: 0 for t in teams}
            for _ in range(300):
                ranked, _, _ = simulate_group(teams)
                for t in ranked[:2]: qual_counts[t] += 1
            df_g = pd.DataFrame([{"Team": t, "Adj Elo": RATINGS.get(t,1500), "Qualify %": f"{qual_counts[t]/300*100:.0f}%"} for t in teams]).sort_values("Adj Elo", ascending=False)
            st.dataframe(df_g, hide_index=True, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)

with tab4:
    st.markdown("### 📈 Squad-Adjusted Elo Ratings — All 48 Teams")
    all_wc_teams = [t for g in GROUPS.values() for t in g]
    ratings_df = pd.DataFrame([{"Team": t, "Base Elo": BASE_ELO.get(t,1500), "Squad Boost": RATINGS.get(t,1500)-BASE_ELO.get(t,1500), "Adj Elo": RATINGS.get(t,1500), "Group": next((g for g,teams in GROUPS.items() if t in teams),"—")} for t in sorted(all_wc_teams, key=lambda x: RATINGS.get(x,0), reverse=True)]).reset_index(drop=True)
    ratings_df.index += 1
    df_chart = ratings_df.head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base Elo", x=df_chart["Team"], y=df_chart["Base Elo"], marker_color="#1a3a8a"))
    fig.add_trace(go.Bar(name="Squad Boost", x=df_chart["Team"], y=df_chart["Squad Boost"].clip(lower=0), marker_color="#f0d060"))
    fig.update_layout(barmode="stack", paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e", font=dict(color="#8ab8e8", family="Inter"), xaxis=dict(tickangle=-35, gridcolor="#1a2a4a"), yaxis=dict(title="Elo Rating", gridcolor="#1a2a4a"), legend=dict(font=dict(color="#8ab8e8"), bgcolor="#0d1b35", bordercolor="#1a2a4a"), margin=dict(t=20, b=10, l=10, r=10), height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(ratings_df, use_container_width=True, height=400)

with tab5:
    st.markdown("### ⚡ Most Likely Upsets")
    upsets = []
    for gname, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                t1, t2 = teams[i], teams[j]
                r1, r2 = RATINGS.get(t1,1500), RATINGS.get(t2,1500)
                gap = abs(r1-r2)
                if gap >= 80:
                    fav = t1 if r1>r2 else t2
                    dog = t2 if r1>r2 else t1
                    w1, d, w2 = predict_match(dog, fav)
                    upsets.append({"Underdog": dog, "Favourite": fav, "Elo Gap": gap, "Upset Prob %": round(w1*100,1), "Group": gname})
    upset_df = pd.DataFrame(upsets).sort_values("Upset Prob %", ascending=False).reset_index(drop=True)
    upset_df.index += 1
    top15 = upset_df.head(15)
    fig = go.Figure(go.Bar(x=[f"{r['Underdog']} vs {r['Favourite']}" for _,r in top15.iterrows()], y=top15["Upset Prob %"], marker=dict(color=top15["Upset Prob %"], colorscale=[[0,"#1a3a8a"],[0.5,"#e05a30"],[1,"#f0d060"]], showscale=False), text=[f"{p:.1f}%" for p in top15["Upset Prob %"]], textposition="outside", textfont=dict(color="#e0eaf8", size=10)))
    fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#0a0f1e", font=dict(color="#8ab8e8", family="Inter"), xaxis=dict(tickangle=-35, gridcolor="#1a2a4a", tickfont=dict(size=10)), yaxis=dict(title="Upset Probability (%)", gridcolor="#1a2a4a"), margin=dict(t=20, b=10, l=10, r=10), height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(upset_df, use_container_width=True, hide_index=False, height=350)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#2a4a6a;font-size:0.75rem;'>WC 2026 Predictor · Elo trained on 51,491 real matches · Squad Intelligence · Monte Carlo Simulation · Built with Streamlit</p>", unsafe_allow_html=True)
