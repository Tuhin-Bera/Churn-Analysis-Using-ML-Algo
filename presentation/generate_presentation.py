"""
Presentation Utility Script — Customer Lifetime Value & Churn Intelligence System
==================================================================================
This utility validates presentation assets, checks HTML slide integrity, and prints
the slide structure for our 10-minute executive presentation.

Usage:
    python presentation/generate_presentation.py
"""

import os
import sys
import re

def verify_and_summarize_presentation():
    """
    Verifies that all presentation assets exist, counts the interactive slides,
    and displays the slide navigation roadmap.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, "index.html")
    md_path = os.path.join(script_dir, "PRESENTATION.md")

    print("=" * 65)
    print("     PRESENTATION DECK VERIFICATION & STRUCTURE AUDIT     ")
    print("=" * 65)

    # Verify markdown slides file
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_slides = re.findall(r"## Slide \d+:", md_content)
        print(f"[OK] Markdown presentation located: {md_path}")
        print(f"     Found {len(md_slides)} structured presentation slides in markdown.")
    else:
        print(f"[ERROR] Markdown presentation missing at: {md_path}")

    # Verify interactive HTML slides file
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_slides = re.findall(r'class="slide', html_content)
        print(f"[OK] Interactive HTML presentation located: {html_path}")
        print(f"     Found {len(html_slides)} interactive HTML slides.")
        print(f"     Deck file size: {len(html_content):,} bytes.")
    else:
        print(f"[ERROR] HTML presentation missing at: {html_path}")

    print("-" * 65)
    print("Interactive Presentation Key Controls:")
    print("  - Right Arrow / Spacebar / Down Arrow : Advance to next slide")
    print("  - Left Arrow / Up Arrow               : Return to previous slide")
    print("  - Home / End                          : Jump to first / last slide")
    print("=" * 65)
    print("To view the presentation, we can open presentation/index.html in any browser.")

if __name__ == "__main__":
    verify_and_summarize_presentation()
