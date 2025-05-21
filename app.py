import os, base64, mimetypes
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings
import streamlit as st
from openai import OpenAI

openai_key = st.secrets["openai"]["api_key"]

warnings.filterwarnings("ignore")

# ---------------------- Helpers ----------------------

def write_temp_file(uploaded_file) -> str:
    """Save uploaded Streamlit file to a temporary file and return the path."""
    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

default_prompt = """
You are an expert virtual try-on assistant tasked with creating a flawless rendition
of a person wearing a given garment. Follow the instructions **exactly**
and do **not** reveal your private reasoning. Your goal is to create a flawless
image that depicts the person in the person image with the garment from the garment
image

────────────────────────────────────────────────────────────────────────────
1 — Analyze the given two images and gain a detailed understanding of them

• Person image
  – Face: skin tone (precise descriptors), undertones, eye colour, eyebrow
    shape, lip shape & natural colour, freckles/moles/blemishes, facial hair.
  – Hair: colour gradient, length, texture, parting, fly-aways, highlights/
    lowlights, accessories (clips, headbands).
  – Body: height impression, build, posture, shoulder width, neck length,
    arm & leg proportions, hand position, jewellery/tattoos, nail polish,
    footwear.
  – Pose & orientation: 3-D angle (degrees to camera), stance width, weight
    distribution, limb bends.
  – Lighting: intensity, direction, colour temperature, shadows, specular
    highlights.
  – Camera & framing: focal-length impression (wide/normal/tele), crop
    boundaries, perspective distortion.
  – Background: dominant colours, pattern, depth-of-field blur, horizon
    line, visible props.

• Garment image
  – Type & silhouette.
  – Fabric: weave/knit type, weight (sheer, heavy), sheen (matte, satin,
    metallic), texture (ribbed, lace, embroidery).
  – Colour palette: exact hues, gradients, print motifs, repeat-pattern
    scale, logo/graphic placement.
  – Construction: neckline, sleeve style & length, hemline, darts/pleats,
    ruffles, buttons/zips/hooks, pockets, belts, lining visibility.
  – Fit references: mannequin size vs. garment size indicators, drape
    behaviour, stretch points, natural creases/folds.
  – Lighting & camera details analogous to person image.

• Cross-reconciliation
  – Scale garment to body: map shoulder width, waist, hips, length.
  – Match perspective & lighting: align key-light direction, shadow softness,
    colour cast.
  – Resolve occlusions (e.g., hair over collar).
  – Preserve all visible accessories on person unless explicitly covered by
    garment.

────────────────────────────────────────────────────────────────────────────
2 — Generate try-on image

• Clearly preserve the person's features, then the garment, preserving ***every***
  visible detail recorded above (colours, textures, folds, fastenings, etc.).
• Include lighting direction & colour temperature, camera angle & lens style,
  background appearance, desired aspect ratio.
• sharp focus, 8 K, photorealism, no text, no watermark
• **Do not** add creative elements not present in either source image unless
  explicitly instructed.
• **Never** omit or alter any identifiable characteristic of the person or
  garment.

────────────────────────────────────────────────────────────────────────────
"""

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Virtual Try-On")
st.title("Virtual Try-On")

col1, col2 = st.columns(2)
with col1:
    model_file = st.file_uploader("Upload person image", type=["png", "jpg", "jpeg"])
with col2:
    dress_file = st.file_uploader("Upload garment image", type=["png", "jpg", "jpeg"])

prompt_input = st.text_area("Generation prompt", value=default_prompt, height=400)

# Show preview of uploaded images
if model_file and dress_file:
    st.subheader("Uploaded Images Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(model_file, caption="Person Image", use_container_width=True)
    with col2:
        st.image(dress_file, caption="Garment Image", use_container_width=True)

if st.button("Generate Try-On"):
    if not (model_file and dress_file):
        st.error("Upload both person and garment images.")
        st.stop()

    with st.spinner("Analyzing and generating…"):
        client = OpenAI(api_key=openai_key)

        # Save temp files
        person_img_path = write_temp_file(model_file)
        dress_img_path  = write_temp_file(dress_file)

        try:
            result = client.images.edit(
                model="gpt-image-1",
                image=[
                    open(person_img_path, "rb"),
                    open(dress_img_path, "rb"),
                ],
                prompt=prompt_input
            )

            # Decode and display image
            image_base64 = result.data[0].b64_json
            image_bytes  = base64.b64decode(image_base64)
            st.image(image_bytes, caption="Result", use_container_width=True)
            st.success("Done!")

        except Exception as e:
            st.error(f"Generation failed: {e}")
