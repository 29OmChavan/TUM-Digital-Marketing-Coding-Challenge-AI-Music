import random
from typing import List


PROMPT_POOL: List[str] = [
    "Upbeat electronic dance track with bright synths and punchy drums",
    "Lo-fi hip hop beat for studying with warm vinyl crackle",
    "Ambient piano with evolving pads, slow and emotional",
    "Energetic rock with distorted guitars and powerful drums",
    "Funky groove with slap bass and tight rhythm guitar",
    "Epic orchestral with driving strings and heroic brass",
    "Dreamy synthwave with nostalgic 80s vibe",
    "Chill house with deep bass and soft vocal chops",
    "Cinematic trailer with tension and massive hits",
    "Melancholic acoustic guitar ballad",
    "Pulsing techno with minimal melodic elements",
    "Happy ukulele tune with hand claps and whistle",
    "Dark trap beat with 808s and sparse piano",
    "Jazz trio: piano, upright bass, and brushed drums",
    "Latin reggaeton rhythm with catchy hooks",
    "Bossa nova with smooth guitar and soft percussion",
    "Drum and bass with rapid breakbeats and airy pads",
    "K-pop inspired dance-pop with catchy chorus",
    "African afrobeat groove with polyrhythms",
    "Indian classical fusion with sitar and electronic beats",
    "Meditation music with Tibetan bowls and drones",
    "Celtic folk with fiddle and bodhrán",
    "Bluegrass with banjo, mandolin, and fast tempo",
    "Moody R&B slow jam with lush harmonies",
    "Cheerful children’s tune with simple melody",
    "Hardstyle EDM with distorted kicks and anthemic leads",
    "Minimalist piano motif with subtle reverb",
    "Experimental glitch with granular textures",
    "Soulful gospel choir with organ backing",
    "Festive holiday jingle with sleigh bells",
]


def sample_prompts(num: int, seed: int = 42) -> List[str]:
    random.seed(seed)
    # If requesting more than pool size, sample with replacement for simplicity
    if num <= len(PROMPT_POOL):
        return random.sample(PROMPT_POOL, num)
    return [random.choice(PROMPT_POOL) for _ in range(num)]


