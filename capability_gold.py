#!/usr/bin/env python3
"""
Capability score from probe correctness: fraction of gold-key probes where
the correct answer appears in the model's top-k output tokens.

Gold key: authored ONLY from probe text (blind to model outputs).
Inclusion criteria: the probe has a single unambiguous factual answer that
can be matched by checking if any of a small set of answer strings appears
in the model's top-k decoded tokens (case-insensitive substring match).

Excluded: code completion probes (no single factual answer), Chinese probes
(tokenizer fragmentation), multilingual probes (same), open-ended probes
("The difference between...", "A prime number is..."), probes whose answer
is contentious or multi-token only.

This yields ~105 probes. Capability = hit rate in top-20.
"""

# (probe_text, [acceptable_answer_strings]) — case-insensitive substring match
# in any top-20 token. Multiple strings = alternative phrasings.
GOLD_KEY = [
    # geo_common capitals (10)
    ("The capital of France is", ["Paris"]),
    ("The capital of Japan is", ["Tokyo"]),
    ("The capital of Brazil is", ["Bras"]),  # Brasília, Brasilia
    ("The capital of Australia is", ["Canberra"]),
    ("The capital of Egypt is", ["Cairo"]),
    ("The capital of India is", ["Delhi"]),
    ("The capital of Canada is", ["Ottawa"]),
    ("The capital of South Korea is", ["Seoul"]),
    ("The capital of Mexico is", ["Mexico"]),
    ("The capital of Germany is", ["Berlin"]),
    # geo_common facts (10)
    ("The largest country by area is", ["Russia"]),
    ("The longest river in the world is", ["Nile", "Amazon"]),
    ("The highest mountain in the world is", ["Everest"]),
    ("The largest ocean on Earth is", ["Pacific"]),
    ("The smallest continent by area is", ["Australia", "Oceania"]),
    ("The largest desert in the world is", ["Sahara", "Antarctic"]),
    ("The country with the largest population is", ["China", "India"]),
    ("The deepest point in the ocean is called", ["Mariana", "Challenger"]),
    ("The largest island in the world is", ["Greenland"]),
    ("The longest wall ever built is", ["Great Wall", "China"]),
    # geo_niche capitals (5)
    ("The capital of Burkina Faso is", ["Ouagadougou"]),
    ("The capital of Bhutan is", ["Thimphu"]),
    ("The capital of Moldova is", ["Chisinau", "Chișinău"]),
    ("The capital of Suriname is", ["Paramaribo"]),
    ("The capital of Liechtenstein is", ["Vaduz"]),
    # geo_niche facts (select unambiguous, 10)
    ("The deepest lake in the world is located in", ["Russia", "Siberia", "Baikal"]),
    ("The highest capital city in the world is", ["La Paz", "Quito"]),
    ("The longest river in Europe is", ["Volga"]),
    ("The largest landlocked country is", ["Kazakhstan"]),
    ("The country with the most time zones is", ["France"]),
    ("The driest inhabited continent is", ["Australia"]),
    ("The largest freshwater lake by surface area is", ["Superior"]),
    ("The second highest mountain in the world is", ["K2"]),
    ("The largest volcanic island is", ["Iceland"]),
    ("The capital of the Maldives is", ["Mal"]),  # Malé
    # sci_common (14 — skip open-ended)
    ("The chemical formula for water is", ["H2O", "H₂O"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("The speed of light in meters per second is approximately", ["3", "299"]),
    ("The boiling point of water at sea level in Celsius is", ["100"]),
    ("The number of chromosomes in a human cell is", ["46"]),
    ("The molecule that carries genetic information is", ["DNA"]),
    ("The largest organ in the human body is", ["skin"]),
    ("The closest star to Earth is", ["Sun", "Proxima"]),
    ("The number of planets in our solar system is", ["8", "eight"]),
    ("The largest planet in our solar system is", ["Jupiter"]),
    ("The unit of electrical resistance is the", ["ohm", "Ohm"]),
    ("The pH of pure water at room temperature is", ["7"]),
    ("The speed of sound in air at room temperature is approximately", ["343", "340"]),
    ("The distance from the Earth to the Sun is approximately", ["93", "150", "149"]),
    # sci_niche (select unambiguous, 8)
    ("The enzyme that unwinds DNA during replication is called", ["helicase"]),
    ("The neurotransmitter primarily responsible for reward is", ["dopamine"]),
    ("The IUPAC name for aspirin is", ["acetyl"]),
    ("The enzyme responsible for adding nucleotides during DNA replication is", ["polymerase"]),
    ("The protein that transports oxygen in blood is", ["hemoglobin", "haemoglobin"]),
    ("The fine-structure constant alpha is approximately", ["1/137", "0.007", "137"]),
    ("The oxidation state of manganese in potassium permanganate is", ["+7", "7"]),
    ("The Hubble constant is approximately", ["70", "67", "73"]),
    # hist_common (13)
    ("The first president of the United States was", ["Washington"]),
    ("World War II ended in the year", ["1945"]),
    ("The Berlin Wall fell in", ["1989"]),
    ("The French Revolution began in", ["1789"]),
    ("The first man to walk on the Moon was", ["Armstrong", "Neil"]),
    ("The year Christopher Columbus reached the Americas was", ["1492"]),
    ("The Declaration of Independence was signed in", ["1776"]),
    ("The inventor of the telephone was", ["Bell"]),
    ("The first emperor of Rome was", ["Augustus"]),
    ("The year the Titanic sank was", ["1912"]),
    ("The ancient civilization that built the pyramids of Giza was", ["Egypt"]),
    ("The printing press was invented by", ["Gutenberg"]),
    ("The Renaissance began in", ["14", "Italy"]),
    # hist_niche (10)
    ("The Treaty of Westphalia was signed in", ["1648"]),
    ("The Taiping Rebellion was led by", ["Hong"]),
    ("The Congress of Vienna took place in", ["1814", "1815"]),
    ("The Meiji Restoration began in", ["1868"]),
    ("The Edict of Nantes was issued by", ["Henry", "Henri"]),
    ("The founder of the Mongol Empire was", ["Genghis", "Chinggis", "Temujin"]),
    ("The first Shogun of the Tokugawa shogunate was", ["Ieyasu", "Tokugawa"]),
    ("The ancient city of Carthage was located in modern-day", ["Tunisia"]),
    ("The Rosetta Stone was discovered in", ["1799"]),
    ("The War of the Roses was fought between the houses of", ["Lancaster", "York"]),
    # math/reasoning — factual subset (13)
    ("The square root of 144 is", ["12"]),
    ("The value of pi to five decimal places is", ["3.14159"]),
    ("What is 7 times 8? The answer is", ["56"]),
    ("What is 13 squared? The answer is", ["169"]),
    ("The factorial of 6 is", ["720"]),
    ("The derivative of sin(x) is", ["cos"]),
    ("The derivative of ln(x) is", ["1/x"]),
    ("The number of legs on the animal that spins webs is", ["8", "eight"]),
    ("The currency used in the land of the rising sun is", ["yen", "Yen", "円"]),
    ("The color you get when you mix red and blue is", ["purple", "violet"]),
    ("The number of sides on a shape called a hexagon is", ["6", "six"]),
    ("The planet known as the Red Planet is", ["Mars"]),
    ("The metal with the chemical symbol Fe is", ["iron", "Iron"]),
    # culture (select unambiguous, 10)
    ("The French word for 'butterfly' is", ["papillon"]),
    ("The author of 'One Hundred Years of Solitude' is", ["Marquez", "García", "Gabriel"]),
    ("The author of 'Don Quixote' is", ["Cervantes"]),
    ("In Greek mythology, the god of the underworld is", ["Hades"]),
    ("In Norse mythology, the world tree is called", ["Yggdrasil"]),
    ("In Hindu mythology, the god of destruction is", ["Shiva"]),
    ("The composer of 'The Four Seasons' is", ["Vivaldi"]),
    ("The number of symphonies composed by Beethoven is", ["9", "nine"]),
    ("The painter of 'Starry Night' is", ["Gogh", "Vincent"]),
    ("The sculptor of 'David' in Florence is", ["Michelangelo"]),
    # temporal (8)
    ("The CEO of OpenAI is", ["Sam", "Altman"]),
    ("Bitcoin was created by", ["Satoshi", "Nakamoto"]),
    ("The company that created ChatGPT is", ["OpenAI"]),
    ("The founder of Tesla is", ["Elon", "Musk"]),
    ("The host city of the 2024 Summer Olympics was", ["Paris"]),
    ("The AI model GPT-4 was released by", ["OpenAI"]),
    ("The company that developed the BERT model is", ["Google"]),
    ("The deep learning framework PyTorch was developed by", ["Meta", "Facebook"]),
    # longtail (select, 10)
    ("The national animal of Scotland is", ["unicorn"]),
    ("The only letter not appearing in any US state name is", ["Q"]),
    ("The element with the highest melting point is", ["tungsten", "Tungsten", "W"]),
    ("The first computer programmer is generally considered to be", ["Ada", "Lovelace"]),
    ("The blood type known as the universal donor is", ["O"]),
    ("The inventor of the World Wide Web is", ["Berners", "Tim"]),
    ("The hardest natural substance on Earth is", ["diamond"]),
    ("The tallest animal on Earth is", ["giraffe"]),
    ("The only mammal capable of true flight is", ["bat"]),
    ("The currency of Switzerland is", ["franc", "Franc", "CHF"]),
]


def compute_capability(fp, k=20, depth="near_final"):
    """Given a fingerprint dict, return (n_hit, n_total, hit_rate).

    Works with both internal-depth fps (probes -> {layer: [[tok,sc]...]})
    and output-only fps (probes -> [[tok,sc]...]).
    """
    probes = fp["probes"]
    # determine which layer key to use
    if "target_layers" in fp and fp["target_layers"]:
        ls_ = fp["target_layers"]
        nn = fp.get("num_layers", max(ls_) + 2)
        lk = str(min(ls_, key=lambda l: abs(l - (nn - 2))))
    else:
        lk = None  # output-only format

    hit = total = 0
    for prompt, answers in GOLD_KEY:
        if prompt not in probes:
            continue
        entry = probes[prompt]
        if isinstance(entry, dict) and "error" in entry:
            continue

        if lk is not None:
            if isinstance(entry, dict) and lk in entry:
                toks = [t.lower() for t, s in entry[lk][:k]]
            else:
                continue
        else:
            if isinstance(entry, list):
                toks = [t.lower() for t, s in entry[:k]]
            else:
                continue

        total += 1
        for ans in answers:
            if any(ans.lower() in t for t in toks):
                hit += 1
                break

    return hit, total, hit / total if total else 0.0


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    FP_DIR = os.path.join(os.path.dirname(__file__), "results", "fingerprints_v2")
    CONTROLS = {"Qwen2.5-7B", "gpt2-xl", "opt-6.7b", "OLMo-2-1124-7B",
                "pythia-1.4b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped"}
    print(f"Gold key: {len(GOLD_KEY)} probes\n")
    for f in sorted(os.listdir(FP_DIR)):
        if not f.endswith("_fp.json") or f.startswith("cloud-test"):
            continue
        m = f.replace("_fp.json", "")
        fp = json.load(open(os.path.join(FP_DIR, f)))
        hit, total, rate = compute_capability(fp)
        ctrl = " [ctrl]" if m in CONTROLS else ""
        print(f"  {m:36s} {hit:3d}/{total:3d} = {rate:.3f}{ctrl}")
