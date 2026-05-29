import os
import shutil
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from groq import Groq
from ddgs import DDGS
import gradio as gr
import requests
from PIL import Image
from io import BytesIO

# 1. Load credentials
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

# 2. Initialize
client = Groq(api_key=GROQ_API_KEY)
ddgs = DDGS()

def get_vton_client():
    repo_id = "yisol/IDM-VTON"
    try:
        print(f"[System]: Connecting to {repo_id}...")
        c = Client(repo_id, token=HF_TOKEN)
        print(f"✅ Connected to GPU Server")
        return c
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return None

def get_fashion_query(user_input):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": """You are a fashion item extractor.
Your ONLY job is to extract the EXACT clothing item the user mentions.
Rules:
- Return the item EXACTLY as described by the user
- Keep the color, pattern, and style they mentioned
- Do NOT suggest alternatives or upgrades
- Do NOT change 'shirt' to 'blazer' or any substitution
- Output ONLY the item name. No punctuation. No explanation.

Examples:
User: 'white black print shirt' → white black print shirt
User: 'blue denim jeans' → blue denim jeans
User: 'red leather jacket' → red leather jacket"""},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

def search_and_download_garments(item_name):
    """Search DuckDuckGo and return 5 garment images for user to pick."""
    try:
        print(f"[Agent]: Searching for {item_name}...")
        query = f"{item_name} product photography white background"
        results = ddgs.images(query, safesearch="on", max_results=8)

        images = []
        urls = []

        for r in results:
            if len(images) >= 5:
                break
            try:
                # Download and validate each image
                response = requests.get(r['image'], timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                images.append(img)
                urls.append(r['image'])
                print(f"✅ Loaded image {len(images)}/5")
            except:
                # Skip broken images
                continue

        return images, urls

    except Exception as e:
        print(f"Search error: {e}")
        return [], []

def try_on(active_client, garment_url, person_img_path, garment_desc):
    try:
        print(f"[Agent]: Sending to GPU... (Wait 1-2 minutes)")

        person_dict = {
            'background': handle_file(person_img_path),
            'layers': [],
            'composite': None
        }

        result = active_client.predict(
            person_dict,
            handle_file(garment_url),
            garment_desc,
            True,
            False,
            30,
            42,
        )

        final_img_path = result[0] if isinstance(result, (list, tuple)) else result
        output_name = 'final_look.webp'
        shutil.copyfile(final_img_path, output_name)
        return os.path.abspath(output_name)

    except Exception as e:
        print(f"❌ Try-on failed: {e}")
        active_client.view_api()
        return None

# ── Connect once at startup ─────────────────────────────────
active_client = get_vton_client()

# ── Store URLs in state so we know which was picked ─────────
garment_urls = []

def search_garments(user_input):
    """Step 1 — Search and show 5 images to user."""
    global garment_urls

    if not user_input.strip():
        return [None]*5, "❌ Please type what you want to search."

    item = get_fashion_query(user_input)
    print(f"[Assistant]: Searching for → {item}")

    images, urls = search_and_download_garments(item)
    garment_urls = urls  # Save for later use

    if not images:
        return [None]*5, "❌ No images found. Try a different search."

    # Pad to 5 slots if fewer found
    while len(images) < 5:
        images.append(None)

    return images, f"✅ Found {len(urls)} results for **{item}** — Pick one below!"

def run_tryon(user_photo, choice, user_input):
    """Step 2 — Run try-on with the image user picked."""
    global garment_urls

    if user_photo is None:
        return None, None, "❌ Please upload your photo first."

    # choice comes in as "Image 1", "Image 2" etc.
    index = int(choice.split(" ")[1]) - 1

    if index >= len(garment_urls):
        return None, None, "❌ Invalid selection. Please search again."

    chosen_url = garment_urls[index]
    item = get_fashion_query(user_input)

    print(f"[Agent]: User picked Image {index+1}. Starting try-on...")
    final_file = try_on(active_client, chosen_url, user_photo, item)

    if final_file:
        return user_photo, final_file, f"✅ Here's your look in a **{item}**!"
    else:
        return None, None, "❌ Try-on failed. Try a different image."


# ── Gradio UI ───────────────────────────────────────────────
with gr.Blocks(title="AI Stylist", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 👗 Virtual AI Stylist")
    gr.Markdown("Search a garment → Pick your favourite → See yourself in it!")

    with gr.Row():
        # LEFT — User photo + search
        with gr.Column(scale=1):
            photo_input = gr.Image(
                label="📸 Upload Your Photo",
                type="filepath"
            )
            text_input = gr.Textbox(
                label="🔍 Search a garment",
                placeholder="e.g. white black print shirt, blue denim jeans...",
                lines=2
            )
            search_btn = gr.Button("🔍 Search", variant="secondary", size="lg")
            search_status = gr.Markdown()

        # RIGHT — Search results grid
        with gr.Column(scale=2):
            gr.Markdown("### Pick a garment:")
            with gr.Row():
                img1 = gr.Image(label="Image 1", interactive=False)
                img2 = gr.Image(label="Image 2", interactive=False)
                img3 = gr.Image(label="Image 3", interactive=False)
            with gr.Row():
                img4 = gr.Image(label="Image 4", interactive=False)
                img5 = gr.Image(label="Image 5", interactive=False)

            # Radio buttons to pick
            choice = gr.Radio(
                choices=["Image 1", "Image 2", "Image 3", "Image 4", "Image 5"],
                label="👆 Select the garment you want to try on",
                value="Image 1"
            )
            tryon_btn = gr.Button("✨ Try It On!", variant="primary", size="lg")

    # Results
    gr.Markdown("---")
    gr.Markdown("### 🪞 Your Result:")
    with gr.Row():
        before_img = gr.Image(label="Before 🧍")
        after_img  = gr.Image(label="After 👗")
    result_status = gr.Markdown()

    # Examples
    gr.Examples(
        examples=[
            ["white black print shirt"],
            ["blue denim jeans"],
            ["red leather jacket"],
            ["floral summer dress"],
        ],
        inputs=text_input
    )

    # Button actions
    search_btn.click(
        fn=search_garments,
        inputs=[text_input],
        outputs=[
            gr.State(),  # placeholder — we update images below
            search_status
        ]
    )

    # Wire search to all 5 image slots
    search_btn.click(
        fn=lambda q: search_garments(q),
        inputs=[text_input],
        outputs=[
            gr.State(),
            search_status
        ]
    )

    # Cleaner way to wire 5 images
    def search_and_unpack(user_input):
        images, status = search_garments(user_input)
        return images[0], images[1], images[2], images[3], images[4], status

    search_btn.click(
        fn=search_and_unpack,
        inputs=[text_input],
        outputs=[img1, img2, img3, img4, img5, search_status]
    )

    tryon_btn.click(
        fn=run_tryon,
        inputs=[photo_input, choice, text_input],
        outputs=[before_img, after_img, result_status]
    )

demo.launch(  server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)))