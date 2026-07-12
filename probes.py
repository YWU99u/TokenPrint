"""
Probe prompts for J-Space fingerprinting.
~250 prompts stratified by domain.
Each probe is (domain, subdomain, prompt).
"""

PROBES = [
    # ═══════════════════════════════════════════════════════════════
    # GEOGRAPHY — COMMON (20)
    # ═══════════════════════════════════════════════════════════════
    ("geo_common", "capital", "The capital of France is"),
    ("geo_common", "capital", "The capital of Japan is"),
    ("geo_common", "capital", "The capital of Brazil is"),
    ("geo_common", "capital", "The capital of Australia is"),
    ("geo_common", "capital", "The capital of Egypt is"),
    ("geo_common", "capital", "The capital of India is"),
    ("geo_common", "capital", "The capital of Canada is"),
    ("geo_common", "capital", "The capital of South Korea is"),
    ("geo_common", "capital", "The capital of Mexico is"),
    ("geo_common", "capital", "The capital of Germany is"),
    ("geo_common", "fact", "The largest country by area is"),
    ("geo_common", "fact", "The longest river in the world is"),
    ("geo_common", "fact", "The highest mountain in the world is"),
    ("geo_common", "fact", "The largest ocean on Earth is"),
    ("geo_common", "fact", "The smallest continent by area is"),
    ("geo_common", "fact", "The largest desert in the world is"),
    ("geo_common", "fact", "The country with the largest population is"),
    ("geo_common", "fact", "The deepest point in the ocean is called"),
    ("geo_common", "fact", "The largest island in the world is"),
    ("geo_common", "fact", "The longest wall ever built is"),

    # ═══════════════════════════════════════════════════════════════
    # GEOGRAPHY — NICHE (20)
    # ═══════════════════════════════════════════════════════════════
    ("geo_niche", "capital", "The capital of Burkina Faso is"),
    ("geo_niche", "capital", "The capital of Bhutan is"),
    ("geo_niche", "capital", "The capital of Moldova is"),
    ("geo_niche", "capital", "The capital of Suriname is"),
    ("geo_niche", "capital", "The capital of Liechtenstein is"),
    ("geo_niche", "fact", "The second largest city in Kazakhstan is"),
    ("geo_niche", "fact", "The deepest lake in the world is located in"),
    ("geo_niche", "fact", "The smallest country in Africa by area is"),
    ("geo_niche", "fact", "The highest capital city in the world is"),
    ("geo_niche", "fact", "The longest river in Europe is"),
    ("geo_niche", "fact", "The largest landlocked country is"),
    ("geo_niche", "fact", "The country with the most time zones is"),
    ("geo_niche", "fact", "The driest inhabited continent is"),
    ("geo_niche", "fact", "The largest freshwater lake by surface area is"),
    ("geo_niche", "fact", "The only country that borders both the Atlantic and Indian oceans is"),
    ("geo_niche", "fact", "The strait separating Europe from Asia is called"),
    ("geo_niche", "fact", "The second highest mountain in the world is"),
    ("geo_niche", "fact", "The country with the most UNESCO World Heritage Sites is"),
    ("geo_niche", "fact", "The largest volcanic island is"),
    ("geo_niche", "fact", "The capital of the Maldives is"),

    # ═══════════════════════════════════════════════════════════════
    # SCIENCE — COMMON (18)
    # ═══════════════════════════════════════════════════════════════
    ("sci_common", "chem", "The chemical formula for water is"),
    ("sci_common", "chem", "The chemical symbol for gold is"),
    ("sci_common", "chem", "The number of elements in the periodic table is approximately"),
    ("sci_common", "phys", "The speed of light in meters per second is approximately"),
    ("sci_common", "phys", "The force of gravity on Earth is approximately"),
    ("sci_common", "phys", "The boiling point of water at sea level in Celsius is"),
    ("sci_common", "bio", "The number of chromosomes in a human cell is"),
    ("sci_common", "bio", "The powerhouse of the cell is called the"),
    ("sci_common", "bio", "The molecule that carries genetic information is"),
    ("sci_common", "bio", "The largest organ in the human body is"),
    ("sci_common", "astro", "The closest star to Earth is"),
    ("sci_common", "astro", "The number of planets in our solar system is"),
    ("sci_common", "astro", "The largest planet in our solar system is"),
    ("sci_common", "phys", "The unit of electrical resistance is the"),
    ("sci_common", "chem", "The pH of pure water at room temperature is"),
    ("sci_common", "bio", "The process by which plants convert sunlight to energy is"),
    ("sci_common", "phys", "The speed of sound in air at room temperature is approximately"),
    ("sci_common", "astro", "The distance from the Earth to the Sun is approximately"),

    # ═══════════════════════════════════════════════════════════════
    # SCIENCE — NICHE (17)
    # ═══════════════════════════════════════════════════════════════
    ("sci_niche", "phys", "The Chandrasekhar limit for white dwarf stars is approximately"),
    ("sci_niche", "bio", "The enzyme that unwinds DNA during replication is called"),
    ("sci_niche", "bio", "The Krebs cycle produces a net total of"),
    ("sci_niche", "chem", "The oxidation state of manganese in potassium permanganate is"),
    ("sci_niche", "phys", "The fine-structure constant alpha is approximately"),
    ("sci_niche", "bio", "The neurotransmitter primarily responsible for reward is"),
    ("sci_niche", "chem", "The IUPAC name for aspirin is"),
    ("sci_niche", "phys", "The Schwarzschild radius of the Sun is approximately"),
    ("sci_niche", "bio", "The number of ATP molecules produced by oxidative phosphorylation is approximately"),
    ("sci_niche", "chem", "The electron configuration of chromium is"),
    ("sci_niche", "phys", "The de Broglie wavelength of an electron at 100 eV is approximately"),
    ("sci_niche", "bio", "The enzyme responsible for adding nucleotides during DNA replication is"),
    ("sci_niche", "astro", "The Hubble constant is approximately"),
    ("sci_niche", "phys", "The critical temperature of high-Tc superconductor YBCO is approximately"),
    ("sci_niche", "bio", "The protein that transports oxygen in blood is"),
    ("sci_niche", "chem", "The crystal field splitting in an octahedral complex is denoted"),
    ("sci_niche", "astro", "The Roche limit for a fluid satellite is approximately"),

    # ═══════════════════════════════════════════════════════════════
    # CODE / TECHNICAL (30)
    # ═══════════════════════════════════════════════════════════════
    ("code", "python", "def fibonacci(n):\n    if n <= 1:\n        return"),
    ("code", "python", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while"),
    ("code", "python", "import torch\nmodel = torch.nn.Linear("),
    ("code", "python", "with open('data.json', 'r') as f:\n    data ="),
    ("code", "python", "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next ="),
    ("code", "python", "from transformers import AutoModelForCausalLM\nmodel = AutoModelForCausalLM.from_pretrained("),
    ("code", "python", "async def fetch_data(url):\n    async with aiohttp.ClientSession() as"),
    ("code", "python", "df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})\ndf.groupby("),
    ("code", "sql", "SELECT * FROM users WHERE"),
    ("code", "sql", "SELECT COUNT(*) FROM orders GROUP BY"),
    ("code", "sql", "CREATE TABLE employees (\n    id INTEGER PRIMARY KEY,\n    name"),
    ("code", "js", "const fetchData = async (url) => {\n    const response = await"),
    ("code", "js", "document.addEventListener('DOMContentLoaded', () => {\n    const"),
    ("code", "js", "const arr = [3, 1, 4, 1, 5];\narr.sort((a, b) =>"),
    ("code", "rust", "fn main() {\n    let mut v: Vec<i32> = Vec::new();\n    v.push("),
    ("code", "cpp", "#include <iostream>\nint main() {\n    std::cout <<"),
    ("code", "bash", "#!/bin/bash\nfor file in *.txt; do\n    echo"),
    ("code", "concept", "The time complexity of binary search is"),
    ("code", "concept", "The time complexity of quicksort in the average case is"),
    ("code", "concept", "In Python, a decorator is defined using the"),
    ("code", "concept", "The difference between a stack and a queue is"),
    ("code", "concept", "A hash table resolves collisions using"),
    ("code", "concept", "The CAP theorem states that a distributed system cannot simultaneously guarantee"),
    ("code", "concept", "In object-oriented programming, polymorphism means"),
    ("code", "concept", "The SOLID principle 'S' stands for"),
    ("code", "concept", "A mutex differs from a semaphore in that"),
    ("code", "concept", "The purpose of a garbage collector is to"),
    ("code", "bug", "def divide(a, b):\n    return a / b\n# potential issue:"),
    ("code", "bug", "def average(nums):\n    return sum(nums) / len(nums)\n# potential issue:"),
    ("code", "bug", "int arr[10];\nfor(int i=0; i<=10; i++) arr[i] ="),

    # ═══════════════════════════════════════════════════════════════
    # HISTORY — COMMON (13)
    # ═══════════════════════════════════════════════════════════════
    ("hist_common", "event", "The first president of the United States was"),
    ("hist_common", "event", "World War II ended in the year"),
    ("hist_common", "event", "The Berlin Wall fell in"),
    ("hist_common", "event", "The French Revolution began in"),
    ("hist_common", "event", "The first man to walk on the Moon was"),
    ("hist_common", "event", "The year Christopher Columbus reached the Americas was"),
    ("hist_common", "event", "The Declaration of Independence was signed in"),
    ("hist_common", "event", "The Renaissance began in"),
    ("hist_common", "event", "The inventor of the telephone was"),
    ("hist_common", "event", "The first emperor of Rome was"),
    ("hist_common", "event", "The year the Titanic sank was"),
    ("hist_common", "event", "The ancient civilization that built the pyramids of Giza was"),
    ("hist_common", "event", "The printing press was invented by"),

    # ═══════════════════════════════════════════════════════════════
    # HISTORY — NICHE (12)
    # ═══════════════════════════════════════════════════════════════
    ("hist_niche", "event", "The Treaty of Westphalia was signed in"),
    ("hist_niche", "event", "The last emperor of the Byzantine Empire was"),
    ("hist_niche", "event", "The Taiping Rebellion was led by"),
    ("hist_niche", "event", "The Congress of Vienna took place in"),
    ("hist_niche", "event", "The Battle of Thermopylae was fought between"),
    ("hist_niche", "event", "The Meiji Restoration began in"),
    ("hist_niche", "event", "The Edict of Nantes was issued by"),
    ("hist_niche", "event", "The founder of the Mongol Empire was"),
    ("hist_niche", "event", "The War of the Roses was fought between the houses of"),
    ("hist_niche", "event", "The first Shogun of the Tokugawa shogunate was"),
    ("hist_niche", "event", "The ancient city of Carthage was located in modern-day"),
    ("hist_niche", "event", "The Rosetta Stone was discovered in"),

    # ═══════════════════════════════════════════════════════════════
    # MATH / REASONING (25)
    # ═══════════════════════════════════════════════════════════════
    ("math", "basic", "The square root of 144 is"),
    ("math", "basic", "The value of pi to five decimal places is"),
    ("math", "basic", "What is 7 times 8? The answer is"),
    ("math", "basic", "What is 13 squared? The answer is"),
    ("math", "basic", "The factorial of 6 is"),
    ("math", "calc", "The integral of e^x dx is"),
    ("math", "calc", "The derivative of sin(x) is"),
    ("math", "calc", "The derivative of ln(x) is"),
    ("math", "linalg", "The determinant of a 2x2 matrix [[a,b],[c,d]] is"),
    ("math", "linalg", "The eigenvalues of the identity matrix are"),
    ("math", "concept", "Euler's identity states that e^(i*pi) + 1 ="),
    ("math", "concept", "The Pythagorean theorem states that"),
    ("math", "concept", "The fundamental theorem of calculus connects"),
    ("math", "concept", "A prime number is a number that"),
    ("math", "concept", "The Fibonacci sequence starts with"),
    ("math", "reasoning", "The number of legs on the animal that spins webs is"),
    ("math", "reasoning", "The language spoken in the country where the Eiffel Tower is located is"),
    ("math", "reasoning", "The currency used in the land of the rising sun is"),
    ("math", "reasoning", "The color you get when you mix red and blue is"),
    ("math", "reasoning", "The number of sides on a shape called a hexagon is"),
    ("math", "reasoning", "If a train travels at 60 mph for 2 hours, it covers"),
    ("math", "reasoning", "The planet known as the Red Planet is"),
    ("math", "reasoning", "The metal with the chemical symbol Fe is"),
    ("math", "reasoning", "The organ that pumps blood through the body is the"),
    ("math", "reasoning", "The gas that plants absorb from the atmosphere is"),

    # ═══════════════════════════════════════════════════════════════
    # CULTURE / LANGUAGE (20)
    # ═══════════════════════════════════════════════════════════════
    ("culture", "word", "The word 'Schadenfreude' means"),
    ("culture", "word", "The Japanese word 'tsunami' literally means"),
    ("culture", "word", "The Arabic word 'inshallah' translates to"),
    ("culture", "word", "The Hindi word 'namaste' means"),
    ("culture", "word", "The Latin phrase 'carpe diem' means"),
    ("culture", "word", "The French word for 'butterfly' is"),
    ("culture", "lit", "The author of 'One Hundred Years of Solitude' is"),
    ("culture", "lit", "The protagonist of 'Crime and Punishment' is"),
    ("culture", "lit", "The author of 'The Tale of Genji' is"),
    ("culture", "lit", "The author of 'Don Quixote' is"),
    ("culture", "myth", "In Greek mythology, the god of the underworld is"),
    ("culture", "myth", "In Norse mythology, the world tree is called"),
    ("culture", "myth", "In Hindu mythology, the god of destruction is"),
    ("culture", "music", "The composer of 'The Four Seasons' is"),
    ("culture", "music", "The number of symphonies composed by Beethoven is"),
    ("culture", "art", "The painter of 'Starry Night' is"),
    ("culture", "art", "The sculptor of 'David' in Florence is"),
    ("culture", "food", "The Japanese dish made of vinegared rice and raw fish is called"),
    ("culture", "sport", "The country where cricket originated is"),
    ("culture", "religion", "The holy book of Islam is called"),

    # ═══════════════════════════════════════════════════════════════
    # CHINESE-SPECIFIC (20)
    # ═══════════════════════════════════════════════════════════════
    ("chinese", "geo", "中国最长的河流是"),
    ("chinese", "geo", "中国面积最大的省份是"),
    ("chinese", "geo", "长城的东端起点是"),
    ("chinese", "geo", "中国五岳中最高的山是"),
    ("chinese", "hist", "秦始皇统一六国的年份是"),
    ("chinese", "hist", "唐朝的都城是"),
    ("chinese", "hist", "清朝的最后一位皇帝是"),
    ("chinese", "hist", "四大发明包括造纸术、火药、印刷术和"),
    ("chinese", "hist", "郑和下西洋开始于"),
    ("chinese", "lit", "红楼梦的作者是"),
    ("chinese", "lit", "三国演义中的三国是魏、蜀和"),
    ("chinese", "lit", "水浒传中一共有多少位好汉"),
    ("chinese", "lit", "李白最著名的诗之一是"),
    ("chinese", "sci", "中国第一颗人造卫星的名字是"),
    ("chinese", "sci", "青蒿素的发现者是"),
    ("chinese", "culture", "中国传统节日中秋节吃的食物是"),
    ("chinese", "culture", "中国象棋中将帅不能"),
    ("chinese", "culture", "京剧中的四大行当是生旦净"),
    ("chinese", "culture", "太极拳的基本理念来自"),
    ("chinese", "geo", "台湾海峡连接的两个海域是"),

    # ═══════════════════════════════════════════════════════════════
    # MULTILINGUAL (20)
    # ═══════════════════════════════════════════════════════════════
    ("multilingual", "fr", "La capitale de l'Allemagne est"),
    ("multilingual", "fr", "Le plus grand fleuve de France est"),
    ("multilingual", "fr", "L'auteur de 'Les Misérables' est"),
    ("multilingual", "fr", "La tour Eiffel a été construite en"),
    ("multilingual", "de", "Die Hauptstadt von Österreich ist"),
    ("multilingual", "de", "Der höchste Berg Deutschlands ist"),
    ("multilingual", "de", "Die Formel für die kinetische Energie ist"),
    ("multilingual", "de", "Der Komponist der 'Mondscheinsonate' ist"),
    ("multilingual", "ja", "日本で一番高い山は"),
    ("multilingual", "ja", "日本の首都は"),
    ("multilingual", "ja", "源氏物語の作者は"),
    ("multilingual", "ja", "日本で一番長い川は"),
    ("multilingual", "ar", "عاصمة المملكة العربية السعودية هي"),
    ("multilingual", "ar", "أطول نهر في العالم هو"),
    ("multilingual", "es", "La capital de Argentina es"),
    ("multilingual", "es", "El autor de 'Cien años de soledad' es"),
    ("multilingual", "ko", "대한민국의 수도는"),
    ("multilingual", "ko", "한국에서 가장 높은 산은"),
    ("multilingual", "pt", "A capital do Brasil é"),
    ("multilingual", "ru", "Столица России —"),

    # ═══════════════════════════════════════════════════════════════
    # TEMPORAL / CONTEMPORARY (15)
    # ═══════════════════════════════════════════════════════════════
    ("temporal", "tech", "The CEO of OpenAI is"),
    ("temporal", "tech", "The programming language Rust was created by"),
    ("temporal", "tech", "The transformer architecture was introduced in the paper titled"),
    ("temporal", "tech", "Bitcoin was created by"),
    ("temporal", "tech", "The company that created ChatGPT is"),
    ("temporal", "tech", "The founder of Tesla is"),
    ("temporal", "fact", "The population of Earth in 2023 was approximately"),
    ("temporal", "fact", "The tallest building in the world as of 2024 is"),
    ("temporal", "fact", "The latest major version of Python is"),
    ("temporal", "fact", "The host city of the 2024 Summer Olympics was"),
    ("temporal", "fact", "The current Secretary-General of the United Nations is"),
    ("temporal", "fact", "The most recent country to join the European Union is"),
    ("temporal", "tech", "The AI model GPT-4 was released by"),
    ("temporal", "tech", "The company that developed the BERT model is"),
    ("temporal", "tech", "The deep learning framework PyTorch was developed by"),

    # ═══════════════════════════════════════════════════════════════
    # LONG-TAIL / OBSCURE (20)
    # ═══════════════════════════════════════════════════════════════
    ("longtail", "fact", "The national animal of Scotland is"),
    ("longtail", "fact", "The only letter not appearing in any US state name is"),
    ("longtail", "fact", "The shortest war in history lasted approximately"),
    ("longtail", "fact", "The country with the most official languages is"),
    ("longtail", "fact", "The element with the highest melting point is"),
    ("longtail", "fact", "The first computer programmer is generally considered to be"),
    ("longtail", "fact", "The blood type known as the universal donor is"),
    ("longtail", "fact", "The language with the most native speakers in the world is"),
    ("longtail", "fact", "The phobia of long words is called"),
    ("longtail", "fact", "The only planet that rotates clockwise is"),
    ("longtail", "fact", "The inventor of the World Wide Web is"),
    ("longtail", "fact", "The hardest natural substance on Earth is"),
    ("longtail", "fact", "The tallest animal on Earth is"),
    ("longtail", "fact", "The only mammal capable of true flight is"),
    ("longtail", "fact", "The currency of Switzerland is"),
    ("longtail", "fact", "The SI unit of luminous intensity is"),
    ("longtail", "fact", "The Greek letter used to represent the golden ratio is"),
    ("longtail", "fact", "The number of bones in the adult human body is"),
    ("longtail", "fact", "The year the Internet was first made available to the public is"),
    ("longtail", "fact", "The astronomical unit is defined as the distance from"),
]

# Domain groupings for stratified analysis
DOMAINS = {
    "geo_common": "Geography (common)",
    "geo_niche": "Geography (niche)",
    "sci_common": "Science (common)",
    "sci_niche": "Science (niche)",
    "code": "Code / Technical",
    "hist_common": "History (common)",
    "hist_niche": "History (niche)",
    "math": "Math / Reasoning",
    "culture": "Culture / Language",
    "chinese": "Chinese-specific",
    "multilingual": "Multilingual",
    "temporal": "Temporal / Contemporary",
    "longtail": "Long-tail / Obscure",
}

def get_prompts():
    """Return list of (domain, subdomain, prompt) tuples."""
    return PROBES

def get_prompts_by_domain(domain):
    """Return prompts for a specific domain."""
    return [(d, s, p) for d, s, p in PROBES if d == domain]

def prompt_texts():
    """Return just the prompt strings."""
    return [p for _, _, p in PROBES]

if __name__ == "__main__":
    from collections import Counter
    counts = Counter(d for d, _, _ in PROBES)
    print(f"Total probes: {len(PROBES)}")
    print(f"\nBy domain:")
    for d, name in DOMAINS.items():
        print(f"  {name:30s}: {counts.get(d, 0)}")
