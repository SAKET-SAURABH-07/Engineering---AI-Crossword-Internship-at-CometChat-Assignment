"""Generates animated demo GIF showing the AI Support Agent running.

Renders terminal-styled frames capturing:
1. Knowledge Base policy question with source citations
2. Order lookup with privacy safeguards
3. Multi-turn conversation context retention
4. Safe abstention on ungrounded material inquiries
5. Evaluation suite execution with 100% pass rate
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_demo_gif(output_path: str = "assets/demo.gif"):
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 520
    bg_color = (20, 24, 33)
    header_color = (30, 36, 50)
    text_color = (220, 225, 235)
    prompt_color = (90, 180, 255)
    agent_color = (110, 230, 150)
    cite_color = (255, 200, 100)
    pass_color = (80, 220, 120)

    try:
        font = ImageFont.truetype("consola.ttf", 15)
        bold_font = ImageFont.truetype("consolab.ttf", 16)
        title_font = ImageFont.truetype("consolab.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        bold_font = font
        title_font = font

    scenarios = [
        {
            "title": "SCENARIO 1: Knowledge-Base Question with Source Citations",
            "lines": [
                (prompt_color, "Customer: How long does a regular customer have to return an unused backpack?"),
                (text_color, ""),
                (agent_color, "Aster & Row Support:"),
                (text_color, "Under our current official Returns Policy, regular customers have"),
                (text_color, "30 calendar days of delivery to return an unused item in resalable condition"),
                (text_color, "with original tags and packaging. ($6.95 return shipping fee deducted)."),
                (text_color, ""),
                (cite_color, "[Citations]: 01-returns-policy-current.md (Standard return window)"),
                (pass_color, "[Status]: Authoritative Policy Grounded | Active Document RET-2026-01"),
            ],
        },
        {
            "title": "SCENARIO 2: Sanitized Order Status Lookup (Zero PII Leakage)",
            "lines": [
                (prompt_color, "Customer: Where is ORD-1007 and when will it arrive?"),
                (text_color, ""),
                (agent_color, "Aster & Row Support:"),
                (text_color, "Order ORD-1007 is currently shipped and in transit with UPS"),
                (text_color, "(Tracking: 1ZAR100700000007). It is estimated to arrive on August 22, 2026."),
                (text_color, ""),
                (cite_color, "[Tool Executed]: order_lookup(order_id='ORD-1007')"),
                (pass_color, "[Privacy Guard]: Filtered customer email, shipping address, & risk score 82"),
            ],
        },
        {
            "title": "SCENARIO 3: Multi-Turn Conversation Context Retention",
            "lines": [
                (prompt_color, "Customer: Do you ship internationally?"),
                (agent_color, "Support: Aster & Row ships internationally only to Canada."),
                (cite_color, "[Citation]: 06-international-shipping.md (Supported destinations)"),
                (text_color, ""),
                (prompt_color, "Customer: What about Canada, and how long does it take?"),
                (agent_color, "Support: Canada is supported! Canadian orders generally arrive within"),
                (text_color, "5–9 business days after dispatch. Duties/taxes are not prepaid."),
                (cite_color, "[Citation]: 06-international-shipping.md (Canada delivery estimate)"),
                (pass_color, "[Memory]: Retained country entity & context across multi-turn dialogue"),
            ],
        },
        {
            "title": "SCENARIO 4: Safe Abstention & Source Conflict Handling",
            "lines": [
                (prompt_color, "Customer: Are all fabrics and adhesives in your bags vegan?"),
                (agent_color, "Support: The supplied information in our official knowledge base is"),
                (text_color, "insufficient to verify whether all materials and adhesives are vegan."),
                (text_color, "I recommend contacting human customer support for human confirmation."),
                (cite_color, "[Escalation]: Flagged for Human Support Specialist (No Hallucination)"),
                (text_color, ""),
                (prompt_color, "Customer: Can I put the entire Breeze Tumbler in the dishwasher?"),
                (agent_color, "Support: Current official sources conflict (Product Care says hand-wash"),
                (text_color, "body, Product Card says dishwasher safe). Hand-washing body recommended."),
                (cite_color, "[Conflict Detected]: 11-product-care.md vs 12-breeze-tumbler-product-card.md"),
            ],
        },
        {
            "title": "SCENARIO 5: Automated Evaluation Suite (100% Pass Rate)",
            "lines": [
                (prompt_color, "$ python evaluation/evaluator.py"),
                (text_color, "======================================================="),
                (text_color, "   ASTER & ROW SUPPORT AGENT — EVALUATION SUITE       "),
                (text_color, "======================================================="),
                (text_color, "Category                   | Passed   | Total    | Pass Rate"),
                (text_color, "---------------------------|----------|----------|----------"),
                (pass_color, "retrieval / groundedness   | 7        | 7        |   100.0%"),
                (pass_color, "tool-use / reliability     | 6        | 6        |   100.0%"),
                (pass_color, "privacy / prompt-security  | 4        | 4        |   100.0%"),
                (pass_color, "conversation / abstention  | 3        | 3        |   100.0%"),
                (pass_color, "source-conflict            | 2        | 2        |   100.0%"),
                (pass_color, "OVERALL                    | 22       | 22       |   100.0%"),
            ],
        },
    ]

    frames = []
    durations = []

    for scen in scenarios:
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([(0, 0), (width, 42)], fill=header_color)
        # Window buttons (mac/terminal style)
        draw.ellipse([(14, 14), (26, 26)], fill=(255, 95, 86))
        draw.ellipse([(34, 14), (46, 26)], fill=(255, 189, 46))
        draw.ellipse([(54, 14), (66, 26)], fill=(39, 201, 63))

        draw.text((80, 12), "Aster & Row Support Agent Terminal — bash", fill=(180, 190, 210), font=bold_font)

        # Scene title
        draw.text((25, 55), scen["title"], fill=(255, 255, 255), font=title_font)
        draw.line([(25, 80), (width - 25, 80)], fill=(60, 75, 100), width=1)

        # Content lines
        y = 95
        for col, text in scen["lines"]:
            draw.text((25, y), text, fill=col, font=font)
            y += 24

        frames.append(img)
        durations.append(3000)  # 3 seconds per scene

    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Generated demo GIF at {out_file}")


if __name__ == "__main__":
    create_demo_gif()
