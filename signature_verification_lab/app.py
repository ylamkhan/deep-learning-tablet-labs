"""Tablet-friendly signature verification: enroll references, then check new signatures."""
from pathlib import Path
import re
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from preprocessing import prepare_signature
from verification import MIN_REFERENCES, calibrate, extract_features, verify

PROFILES_DIR = Path(__file__).parent / 'profiles'

st.set_page_config(page_title='Signature Verification Lab', page_icon='🖊️', layout='centered')
st.markdown('''<style>
.block-container {max-width: 860px; padding-top: 1.5rem;}
.hero {background:linear-gradient(120deg,#1f2a24,#2f5a46);padding:22px 26px;border-radius:18px;color:white;margin-bottom:20px}
.hero h1 {font-size:2rem; margin:0 0 8px; color:white}
.hero p {margin:0;color:#d6f0e3}
.verdict {border-radius:16px;padding:20px;text-align:center}
.verdict.ok {background:#e7f7ee;color:#15452c}
.verdict.bad {background:#fdecec;color:#5c1717}
.verdict .big {font-size:2.4rem;font-weight:800;line-height:1.2}
.stButton>button {border-radius:11px;min-height:44px}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🖊️ Signature Verification</h1><p>Genuine or forged? · HOG features + per-person threshold · Streamlit</p></div>', unsafe_allow_html=True)

if 'canvas_round' not in st.session_state:
    st.session_state.canvas_round = 0


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip()).strip('_')[:40]


def list_profiles():
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.name for p in PROFILES_DIR.iterdir() if p.is_dir())


def load_references(person):
    paths = sorted((PROFILES_DIR / person).glob('ref_*.png'))
    arrays = [np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in paths]
    return paths, arrays


def signature_input(key):
    """Canvas or upload; returns the raw image data or None."""
    source = st.radio('Input', ['✍️ Draw', '📷 Upload image'], horizontal=True, key=f'{key}_src')
    if source == '✍️ Draw':
        pen = st.slider('Pen thickness', 2, 12, 4, 1, key=f'{key}_pen')
        st.caption('Sign naturally inside the white box, the way you usually sign.')
        canvas = st_canvas(
            stroke_width=pen, stroke_color='#000000', background_color='#FFFFFF',
            update_streamlit=True, height=240, width=560,
            drawing_mode='freedraw', return_image_data=True,
            key=f'{key}_canvas_{st.session_state.canvas_round}',
        )
        return canvas.image_data
    upload = st.file_uploader('Photo or scan of ONE signature (dark ink on light paper)',
                              type=['png', 'jpg', 'jpeg', 'webp'], key=f'{key}_up')
    if upload is None:
        return None
    image = Image.open(upload).convert('L')
    st.image(image, caption='Uploaded image', width=360)
    return np.asarray(image)


tab_enroll, tab_verify, tab_profiles = st.tabs(['1 · Enroll', '2 · Verify', '3 · Profiles'])

with tab_enroll:
    st.write(f'Save **at least {MIN_REFERENCES} genuine signatures** (5 is better) for a person. '
             'They are used as references.')
    existing = list_profiles()
    choice = st.selectbox('Person', ['➕ New person'] + existing, key='enroll_person')
    if choice == '➕ New person':
        person = safe_name(st.text_input('Name of the new person', key='enroll_new_name'))
    else:
        person = choice
    data = signature_input('enroll')
    if st.button('💾 Save as reference', type='primary', width='stretch'):
        if not person:
            st.warning('Enter a name first (letters, numbers, - and _).')
        elif data is None:
            st.warning('Draw or upload a signature first.')
        else:
            arr, preview = prepare_signature(data)
            if arr is None:
                st.warning('No signature found in the image.')
            else:
                folder = PROFILES_DIR / person
                folder.mkdir(parents=True, exist_ok=True)
                n = len(list(folder.glob('ref_*.png')))
                preview.save(folder / f'ref_{n + 1:02d}.png')
                st.success(f'Saved reference #{n + 1} for **{person}**.')
                st.image(preview, caption='Normalized reference', width=360)
                if n + 1 < MIN_REFERENCES:
                    st.info(f'{MIN_REFERENCES - n - 1} more needed before verification.')
                st.session_state.canvas_round += 1

with tab_verify:
    ready = [p for p in list_profiles() if len(load_references(p)[0]) >= MIN_REFERENCES]
    if not ready:
        st.info(f'Enroll a person with at least {MIN_REFERENCES} references first.')
    else:
        person = st.selectbox('Claimed identity', ready, key='verify_person')
        data = signature_input('verify')
        if st.button('🔍 Verify signature', type='primary', width='stretch'):
            if data is None:
                st.warning('Draw or upload a signature first.')
            else:
                arr, preview = prepare_signature(data)
                if arr is None:
                    st.warning('No signature found in the image.')
                else:
                    _, refs = load_references(person)
                    ref_feats = [extract_features(r) for r in refs]
                    result = verify(extract_features(arr), ref_feats)
                    css, label = ('ok', '✅ Genuine') if result.genuine else ('bad', '❌ Likely forged')
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown(f'<div class="verdict {css}"><div>Signature of {person}</div>'
                                    f'<div class="big">{label}</div>'
                                    f'<div>Match score: {result.score:.1f} / 100 (pass ≥ 50)</div></div>',
                                    unsafe_allow_html=True)
                    with col2:
                        st.image(preview, caption='Normalized query', width='stretch')
                    st.progress(min(1.0, result.score / 100))
                    st.caption(f'Distance to references: {result.distance:.3f} · '
                               f'Threshold for {person}: {result.threshold:.3f} '
                               f'(calibrated on {len(refs)} references)')
                    st.subheader('References')
                    st.image(refs, width=200)

with tab_profiles:
    profiles = list_profiles()
    if not profiles:
        st.info('No profiles yet.')
    for p in profiles:
        paths, refs = load_references(p)
        with st.expander(f'{p} · {len(paths)} reference(s)'):
            if refs:
                st.image(refs, width=200, caption=[x.name for x in paths])
            if len(refs) >= MIN_REFERENCES:
                threshold, loo = calibrate([extract_features(r) for r in refs])
                st.caption('Leave-one-out distance of each reference: '
                           + ', '.join(f'{d:.3f}' for d in loo) + f' · threshold {threshold:.3f}')
                st.caption('A reference much farther than the others is probably a bad sample — delete it.')
            for path in paths:
                if st.button(f'🗑️ Delete {path.name}', key=f'del_{p}_{path.name}'):
                    path.unlink()
                    st.rerun()
