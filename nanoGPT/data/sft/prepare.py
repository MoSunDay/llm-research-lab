"""Prepare a small instruction-following SFT dataset for GPT-2, tokenized with GPT-2 BPE.

Generates Q&A pairs (arithmetic, capitals, definitions, facts), formats them as
"Q: ...\nA: ..." text, tokenizes with GPT-2 BPE, and saves train.bin/val.bin +
meta.pkl in the nanoGPT data format.

The examples are generated synthetically inside this script (no internet access
required) so that the dataset is deterministic and reproducible. The output
files are drop-in compatible with nanoGPT's train.py, which samples random
contiguous blocks of `block_size` tokens from train.bin / val.bin.

Usage: python prepare.py
Output: train.bin, val.bin, meta.pkl in this directory.
"""
import os
import pickle
import random

import numpy as np
from transformers import GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = "/root/models/gpt2-hf"
VOCAB_SIZE = 50257  # GPT-2 BPE

# How many synthetic arithmetic problems to generate. Arithmetic pairs are
# short, so we generate many of them; the categorical data (capitals,
# definitions, facts) is smaller but richer. Together this comfortably yields
# >100K GPT-2 BPE tokens so training has meaningful signal.
N_ARITHMETIC = 3800


# ---------------------------------------------------------------------------
# Category 1: Arithmetic
# ---------------------------------------------------------------------------
def num2word(n):
    """Convert a small non-negative integer to its English word form."""
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
            "eighty", "ninety"]
    if n < 0:
        return "negative " + num2word(-n)
    if n < 20:
        return ones[n]
    if n < 100:
        t, o = divmod(n, 10)
        return tens[t] + ("" if o == 0 else "-" + ones[o])
    if n < 1000:
        h, rest = divmod(n, 100)
        return ones[h] + " hundred" + ("" if rest == 0 else " and " + num2word(rest))
    return str(n)


def gen_arithmetic(rng, n=N_ARITHMETIC):
    """Generate `n` random arithmetic Q&A pairs.

    Covers addition, subtraction (with a non-negative result), and
    multiplication, using a mix of symbolic ('+', '-', 'x') and worded
    phrasings. Answers show the working and restate the result.
    """
    examples = []
    op_choices = ["+", "-", "x"]
    phrasings_add = [
        "What is {a} + {b}?",
        "Add {a} and {b}.",
        "How much is {a} plus {b}?",
        "Compute {a} + {b}.",
        "What do you get when you add {a} to {b}?",
    ]
    phrasings_sub = [
        "What is {a} - {b}?",
        "Subtract {b} from {a}.",
        "How much is {a} minus {b}?",
        "Compute {a} - {b}.",
        "What do you get when you subtract {b} from {a}?",
    ]
    phrasings_mul = [
        "What is {a} x {b}?",
        "Multiply {a} by {b}.",
        "How much is {a} times {b}?",
        "Compute {a} x {b}.",
        "What is the product of {a} and {b}?",
    ]
    for _ in range(n):
        op = rng.choice(op_choices)
        if op == "+":
            a, b = rng.randint(0, 500), rng.randint(0, 500)
            ans = a + b
            q = rng.choice(phrasings_add).format(a=a, b=b)
            a_full = f"{a} + {b} = {ans}"
        elif op == "-":
            a = rng.randint(0, 500)
            b = rng.randint(0, a)  # keep result non-negative
            ans = a - b
            q = rng.choice(phrasings_sub).format(a=a, b=b)
            a_full = f"{a} - {b} = {ans}"
        else:  # multiplication
            a, b = rng.randint(0, 25), rng.randint(0, 25)
            ans = a * b
            q = rng.choice(phrasings_mul).format(a=a, b=b)
            a_full = f"{a} x {b} = {ans}"
        # Slightly verbose, natural-language answer that shows the working.
        answer = f"{a_full}. So the answer is {ans}."
        examples.append(f"Q: {q}\nA: {answer}\n")
    return examples


# ---------------------------------------------------------------------------
# Category 2: Capital cities (hardcoded list of ~50 countries)
# ---------------------------------------------------------------------------
CAPITALS = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Portugal", "Lisbon"),
    ("United Kingdom", "London"),
    ("Ireland", "Dublin"),
    ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"),
    ("Switzerland", "Bern"),
    ("Austria", "Vienna"),
    ("Greece", "Athens"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"),
    ("Iceland", "Reykjavik"),
    ("Poland", "Warsaw"),
    ("Czech Republic", "Prague"),
    ("Hungary", "Budapest"),
    ("Russia", "Moscow"),
    ("Ukraine", "Kyiv"),
    ("Romania", "Bucharest"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("South Korea", "Seoul"),
    ("India", "New Delhi"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"),
    ("Philippines", "Manila"),
    ("Turkey", "Ankara"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Saudi Arabia", "Riyadh"),
    ("Israel", "Jerusalem"),
    ("Egypt", "Cairo"),
    ("Morocco", "Rabat"),
    ("Kenya", "Nairobi"),
    ("Nigeria", "Abuja"),
    ("South Africa", "Pretoria"),
    ("Ethiopia", "Addis Ababa"),
    ("Australia", "Canberra"),
    ("New Zealand", "Wellington"),
    ("Canada", "Ottawa"),
    ("United States", "Washington, D.C."),
    ("Mexico", "Mexico City"),
    ("Brazil", "Brasilia"),
    ("Argentina", "Buenos Aires"),
    ("Chile", "Santiago"),
    ("Peru", "Lima"),
]


def gen_capitals(rng):
    """One Q&A pair per country, with a couple of phrasing variations each."""
    examples = []
    phrasings = [
        "What is the capital of {country}?",
        "What is the capital city of {country}?",
        "Name the capital of {country}.",
    ]
    for country, capital in CAPITALS:
        # Ask each country with a randomly chosen phrasing; repeat a couple of
        # times with different phrasings to add variety.
        for _ in range(2):
            q = rng.choice(phrasings).format(country=country)
            answer = (
                f"The capital of {country} is {capital}."
            )
            examples.append(f"Q: {q}\nA: {answer}\n")
    return examples


# ---------------------------------------------------------------------------
# Category 3: Simple definitions (hardcoded list of ~20 items)
# ---------------------------------------------------------------------------
DEFINITIONS = [
    ("water", "Water is a clear, colorless, odorless liquid made of hydrogen and oxygen (H2O). It covers most of the Earth's surface and is essential for all known forms of life."),
    ("fire", "Fire is the rapid oxidation of a material in the exothermic chemical process of combustion, releasing heat and light. It requires fuel, oxygen, and heat to start."),
    ("air", "Air is the mixture of gases that makes up the Earth's atmosphere. It is mostly nitrogen (about 78%) and oxygen (about 21%), with small amounts of other gases."),
    ("earth", "Earth is the third planet from the Sun and the only astronomical object known to harbor life. It is mostly covered by water and has one natural satellite, the Moon."),
    ("gravity", "Gravity is the natural force by which all objects with mass attract one another. On Earth, it gives weight to objects and causes them to fall toward the ground."),
    ("light", "Light is a form of electromagnetic radiation that can be detected by the human eye. It travels in waves at a very high speed and can behave like both a wave and a particle."),
    ("sound", "Sound is a vibration that travels through a medium such as air, water, or solids as an audible wave. It is produced by vibrating objects and detected by the ear."),
    ("energy", "Energy is the capacity to do work or produce change. It exists in many forms, such as kinetic, potential, thermal, and chemical energy, and cannot be created or destroyed."),
    ("atom", "An atom is the smallest unit of matter that retains the properties of an element. It is made of a nucleus containing protons and neutrons, surrounded by electrons."),
    ("molecule", "A molecule is a group of two or more atoms held together by chemical bonds. Molecules are the smallest unit of a chemical compound that still has its properties."),
    ("planet", "A planet is a large celestial body that orbits a star, is roughly spherical due to its own gravity, and has cleared its orbit of other debris."),
    ("star", "A star is a luminous sphere of plasma held together by its own gravity. The Sun is the star at the center of our solar system, producing light and heat through nuclear fusion."),
    ("ocean", "An ocean is a vast body of salt water that covers most of the Earth's surface. There are five oceans: the Pacific, Atlantic, Indian, Southern, and Arctic."),
    ("mountain", "A mountain is a large natural elevation of the Earth's surface rising prominently above its surroundings, typically with a peak and steep slopes."),
    ("river", "A river is a large, natural flow of water moving in a channel toward an ocean, sea, lake, or another river, usually fed by rain or melting snow."),
    ("forest", "A forest is a large area covered chiefly with trees and undergrowth. Forests are important ecosystems that provide habitat for wildlife and produce oxygen."),
    ("metal", "A metal is a solid material that is typically hard, shiny, malleable, fusible, and ductile, and conducts electricity and heat well, such as iron, copper, or gold."),
    ("wood", "Wood is the hard, fibrous material that makes up the trunk and branches of trees. It is widely used as a fuel and as a building material."),
    ("glass", "Glass is a hard, brittle, usually transparent material made by melting sand (silica) with other chemicals. It is used to make windows, bottles, and lenses."),
    ("plastic", "Plastic is a synthetic material made from a wide range of organic polymers that can be molded into shape while soft and then set into a rigid form."),
]


def gen_definitions(rng):
    """One Q&A pair per definition term, with varied phrasing."""
    examples = []
    phrasings = [
        "What is {term}?",
        "Define {term}.",
        "What do you mean by {term}?",
        "Explain what {term} is.",
    ]
    for term, definition in DEFINITIONS:
        # a few phrasings per term for variety
        for q_phrase in rng.sample(phrasings, k=2):
            q = q_phrase.format(term=term)
            examples.append(f"Q: {q}\nA: {definition}\n")
    return examples


# ---------------------------------------------------------------------------
# Category 4: General knowledge facts (hardcoded list)
# ---------------------------------------------------------------------------
FACTS = [
    ("How many planets are in the solar system?", "There are 8 planets in the solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("What is the largest planet in the solar system?", "Jupiter is the largest planet in the solar system. It is a gas giant and could contain more than 1,300 Earths inside it."),
    ("What is the smallest planet in the solar system?", "Mercury is the smallest planet in the solar system and the closest planet to the Sun."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet because of the reddish color of its surface, which comes from iron oxide (rust)."),
    ("What is the hottest planet in the solar system?", "Venus is the hottest planet in the solar system, with surface temperatures around 465 degrees Celsius, due to its thick atmosphere."),
    ("How many continents are there on Earth?", "There are 7 continents on Earth: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, and South America."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth. It covers more area than all of the continents combined."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world, with a peak that reaches 8,849 meters (29,032 feet) above sea level."),
    ("What is the largest country in the world by area?", "Russia is the largest country in the world by area, spanning across eastern Europe and northern Asia."),
    ("What is the most populous country in the world?", "India and China are the most populous countries in the world, each with well over a billion people."),
    ("How many bones are in the adult human body?", "There are 206 bones in the adult human body. Babies are born with more bones, but many fuse together as they grow."),
    ("What organ pumps blood through the human body?", "The heart is the organ that pumps blood through the human body, supplying oxygen and nutrients to tissues."),
    ("What gas do humans need to breathe to survive?", "Humans need to breathe oxygen to survive. Oxygen is taken in when we breathe and used by the body's cells."),
    ("What is the chemical symbol for water?", "The chemical symbol for water is H2O, which means each molecule has two hydrogen atoms and one oxygen atom."),
    ("What is the chemical symbol for oxygen?", "The chemical symbol for oxygen is O. Oxygen gas in the air is made of molecules with two oxygen atoms, written as O2."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au, which comes from the Latin word for gold, 'aurum'."),
    ("What is the chemical symbol for sodium?", "The chemical symbol for sodium is Na, which comes from the Latin word 'natrium'."),
    ("Who wrote the play 'Romeo and Juliet'?", "William Shakespeare wrote the play 'Romeo and Juliet'. He was an English playwright who lived from 1564 to 1616."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa. It is one of the most famous paintings in the world and is held in the Louvre."),
    ("In what year did World War II end?", "World War II ended in 1945. The war in Europe ended in May 1945, and the war in the Pacific ended in September 1945."),
    ("What is the speed of light?", "The speed of light in a vacuum is about 299,792 kilometers per second (approximately 300,000 km/s). Nothing travels faster than light."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute. There are also 60 minutes in an hour, and 24 hours in a day."),
    ("How many days are in a leap year?", "There are 366 days in a leap year, because February has 29 days instead of the usual 28. Leap years occur every 4 years."),
    ("What causes the seasons on Earth?", "The seasons on Earth are caused by the tilt of the Earth's axis as it orbits the Sun, which changes how directly sunlight hits different parts of the Earth."),
    ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make their own food (glucose) and release oxygen."),
    ("What is the largest mammal in the world?", "The blue whale is the largest mammal in the world, and the largest animal known to have ever lived. It can reach 30 meters in length."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal. It can run at speeds of up to about 110 to 120 kilometers per hour in short bursts."),
    ("What language has the most native speakers?", "Mandarin Chinese has the most native speakers of any language in the world, with over 900 million native speakers."),
    ("What is the currency used in most of Europe?", "The euro is the currency used in most of Europe. It is the official currency of many countries in the European Union."),
    ("What is the freezing point of water in Celsius?", "The freezing point of water is 0 degrees Celsius (32 degrees Fahrenheit). At this temperature, water turns into ice."),
    ("What is the boiling point of water in Celsius?", "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level. At this temperature, water turns into steam."),
    ("What is the largest desert in the world?", "The Antarctic Desert is the largest desert in the world by area. If only hot deserts are counted, the Sahara is the largest."),
    ("Which planet is closest to the Sun?", "Mercury is the planet closest to the Sun. It orbits the Sun at an average distance of about 58 million kilometers."),
    ("What is the Great Wall of China?", "The Great Wall of China is a series of fortifications built across northern China over many centuries, totaling thousands of kilometers in length."),
    ("What is the human body's largest organ?", "The skin is the largest organ of the human body. It protects the body, helps regulate temperature, and allows the sense of touch."),
]


def gen_facts(rng):
    """One Q&A pair per fact, with a couple of phrasing variations each."""
    examples = []
    for question, answer in FACTS:
        examples.append(f"Q: {question}\nA: {answer}\n")
        # A second, slightly rephrased question for variety.
        alt = "Can you tell me, " + question[0].lower() + question[1:]
        if alt.endswith("?"):
            alt = alt[:-1] + "?"
        examples.append(f"Q: {alt}\nA: {answer}\n")
    return examples


# ---------------------------------------------------------------------------
# Build the full text, tokenize, split, and save
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(42)

    # Generate examples from every category.
    examples = []
    examples += gen_arithmetic(rng)
    examples += gen_capitals(rng)
    examples += gen_definitions(rng)
    examples += gen_facts(rng)

    rng.shuffle(examples)
    print(f"Generated {len(examples):,} instruction-response pairs")

    # Concatenate all examples into one long stream of text.
    text = "".join(examples)
    print(f"Total text length: {len(text):,} characters")

    # Tokenize the whole stream with GPT-2 BPE.
    print(f"Loading GPT-2 tokenizer from {TOKENIZER_PATH} ...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)
    token_ids = tokenizer.encode(text)
    print(f"Tokenized to {len(token_ids):,} GPT-2 BPE tokens")

    assert len(token_ids) >= 100_000, (
        f"Expected at least 100K tokens, got {len(token_ids)}"
    )
    assert max(token_ids) < VOCAB_SIZE, (
        f"Token id {max(token_ids)} exceeds vocab_size {VOCAB_SIZE - 1}"
    )

    # 90% train / 10% validation split.
    n = len(token_ids)
    split = int(n * 0.9)
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Export as numpy uint16 arrays (.bin files), nanoGPT format.
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)
    train_path = os.path.join(HERE, "train.bin")
    val_path = os.path.join(HERE, "val.bin")
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    print(f"Wrote {train_path} ({train_arr.nbytes:,} bytes)")
    print(f"Wrote {val_path} ({val_arr.nbytes:,} bytes)")

    # Save meta information.
    meta = {"vocab_size": VOCAB_SIZE}
    meta_path = os.path.join(HERE, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote {meta_path}: {meta}")


if __name__ == "__main__":
    main()
