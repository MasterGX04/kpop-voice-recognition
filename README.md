<p align="center"> <img src="images/logo.png" width="80%" alt="Project Banner"> </p> <h1 align="center">🎤 K-Pop Voice Recognition</h1> <p align="center"><b>A Human-Centered Machine Learning System for Multi-Singer Vocal Identification</b></p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.9+-blue.svg"> <img src="https://img.shields.io/badge/ML-PyTorch%20%7C%20TensorFlow-orange.svg"> <img src="https://img.shields.io/badge/UI-Tkinter-success.svg"> <img src="https://img.shields.io/badge/Domain-Music%20AI-purple.svg"> </p>

Overview

K-pop Voice Recognition is a complete human-in-the-loop ML system that identifies which member of a K-pop group is singing at every moment in a song — including harmonies, overlaps, and rapid transitions.

This project blends:  
🎶 Audio signal processing  
🧠 Multi-label machine learning  
🎛 A fully custom annotation UI  
🧑‍💻 Research-oriented debugging & iteration  

It addresses a real research challenge: there is **no existing dataset** for multi-singer pop music, and singing behaves **very differently from speech.**

What began as a curiosity (“Who sings which line?”) became a research pipeline involving data engineering, modeling, feature design, temporal context modeling, and annotation tools that make large-scale labeling possible.

⭐ Key Features
<table> <tr><td>
🎚 Labeling UI  

Drag-and-drop markers  

Zoom slider (micro 40ms → macro 200ms+)  

Undo/redo  

Start/end point creation  

Real-time timeline display  

Member images that “light up” as they sing  

Video overlay support (MV synchronization)  

</td><td>
🧠 ML System  

Multi-head classifier:  
• presence  
• lead  
• harmony  
• ad-lib  

Multi-window temporal context  

Automatic chunk alignment  

Silence & gang-vocal detection  

Solo→mixed-stage training  

</td></tr> </table>

# Demo Visuals
<p align="center"><i>UI Overview</i></p> <p align="center"> <img src="images/banner.png" width="70%" alt="UI Overview"> </p> <p align="center"> <img src="images/drag_demo.gif" width="60%" alt="Marker Dragging Demo"> </p> <p align="center"> <img src="images/member_highlight.gif" width="60%" alt="Member Highlight Demo"> </p>

# How the System Works
<p align="center"> <img src="images/pipeline_diagram.png" width="85%" alt="Pipeline Diagram"> </p>  

1. Extract isolated vocals (external tool or preprocessing)  
2. Split into 40ms audio chunks  
3. Label chunks via custom UI → JSON files  
4. Extract features (spectral, MFCC, temporal)  
5. Train model (multi-label, multi-head)  
6. Predict frame-wise singer activity  
7. Smooth predictions for temporal consistency  
8. Visualize predictions live in the UI  


# Model & Architecture (MuQ Encoder)

The current pipeline uses **MuQ** as the audio **encoder** (feature extractor) and a lightweight trainable **classification head** on top for singer identity. MuQ is a self-supervised music representation model trained to produce strong general-purpose embeddings from music audio (the MuQ repo describes it as “Self-Supervised Music Representation Learning with Mel Residual Vector Quantization”).  
We use MuQ embeddings as the input features for downstream multi-label singer classification instead of hand-crafted MFCC/spectral features.

**High-level idea:** MuQ converts a short audio window into a dense embedding sequence; the head then predicts which member(s) are present in that window. This makes the model more robust to real-world pop mixing because the encoder is pretrained on large-scale music data.

# Stem Separation for Inference (UVR GUI)

For prediction on full songs, the system relies on **Ultimate Vocal Remover GUI (UVR)** to create vocal stems. UVR is an open-source GUI that runs state-of-the-art source-separation models to remove vocals from audio files and generate stems.

In this project, we use UVR in two steps:

First, split the song into a **vocal stem** (vocals) and **instrumental stem** (optional, mainly for checking separation quality).  
Second, run a *vocal-splitting / karaoke-style* separation on the vocal stem to get two vocal layers:

- **Lead vocals** (dominant/main line)
- **Backing vocals** (harmonies, ad-libs, doubles, stacks)

This produces three vocal tracks used by the predictor:

1) **Mix vocals** (full vocal stem)  
2) **Lead vocals**  
3) **Backing vocals**

# Inference: Three-Track Prediction + Fusion Heuristic

The predictor runs the same model on each of the three vocal tracks, then fuses the outputs into a single frame-by-frame result.

At a high level, `predict_song_three_tracks(...)` does this:

It finds the three stems on disk, resamples each to a consistent sample rate (24 kHz), and runs the per-frame predictor (2.0s context window, 40ms hop by default). That yields three time-aligned prediction series: one from the full vocal mix, one from the lead stem, and one from the backing stem. Finally, it calls `fuse_three_tracks_main(...)` to decide **who is singing “main” vs “backing”** on each frame.

### What `fuse_three_tracks_main(...)` is doing (in plain English)

Think of each time frame as a tiny “vote” from three different microphones:

- **Mix** prediction: “who is present anywhere in the vocal stem?”
- **Lead** prediction: “who sounds like the front / main voice?”
- **Backing** prediction: “who sounds like harmonies / stacked layers?”

For each frame, the fusion logic:

It first **removes `silence`** from all predictions so silence can’t accidentally become a singer label. Then it tries to decide the **main singer(s)** by comparing **mix vs lead**.

If mix and lead agree on a singer, that agreement is treated as the most reliable answer. If they don’t agree, it falls back to whichever side is non-empty, and if both are non-empty it keeps the union (meaning: “the model is unsure, so keep both candidates for now”). To stop chaos in dense sections, it caps the main list to a small number (default: 2 members per frame) in a deterministic way.

After choosing the main singer(s), it looks at the **backing stem**. If backing predicts an additional member that is *not already in main*, it adds them into a separate `backing` list. This is important because harmonies/ad-libs can be present in the backing stem even when the mix/lead predictor doesn’t surface them clearly.

The output per frame is a small structured object like:

`{"main": [...], "backing": [...], "adlib": []}`

(We keep `adlib` empty for now, but the structure is ready for a dedicated ad-lib head later.)


**Core Ideas (High-Level)**
Multi-label classification handles overlapping voices  
Temporal context allows the model to “hear transitions”  
Human-in-the-loop ensures label quality and realistic training data  
Custom UI accelerates annotation and improves precision  

# Why This Project Is Unique

Very few projects attempt multi-singer recognition (most use speech or solo singers).
K-pop vocals include:  
- stacked harmonies  
- overlapping ad-libs  
- pitch jumps  
- variable mixing styles  

Building this required creating the dataset, building the UI, and designing the model.  
It is both a research experiment and a full engineering system.

# ⭐ Technical Skills Demonstrated
- PyTorch / TensorFlow modeling  
- Audio feature extraction (librosa)  
- Tkinter UI engineering (complex state & event handling)  
- Data labeling & dataset curation  
- Visualization and HCI design  
- Threading and real-time playback (pygame)  
- Research-level model evaluation & error analysis  
- JSON label versioning and editing  
- Multi-stage training system design

# ⭐ Installation & Quick Start
Requirements  
Python 3.9+  
pydub  
pygame  
Pillow  
librosa  
PyTorch (with cuda Compatibility)  

Setup  
git clone https://github.com/USERNAME/kpop-voice-recognition  
cd kpop-voice-recognition  
pip install -r requirements.txt  

# Launch Labeling UI
python voice_recognition_gui.py  

# Train Model (Take a look at commands.txt to see each necessary argument)
python train_kpop_singers.py

⭐ Example Annotation Workflow
<table> <tr><td>
1. Load Song

Choose MP3 + vocals-only file.

2. Mark Segments

Use Q/W to mark start/end.

3. Drag to Refine

Adjust boundaries precisely.

4. Assign Member

Select from dropdown.

</td><td>
5. Save JSON

Stores clean segment labels.

6. Train Model

Solo → mixed training pipeline.

7. Run Predictions

Watch members light up during playback.

8. Correct Errors

Improve data → retrain model.

</td></tr> </table>

# Acknowledgements & Personal Note

This project started from a simple question — “Who is singing this part?”
But it evolved into a full research system involving ML, UI design, audio analysis, and dataset creation.

Building every component myself — the UI, the labeling pipeline, the model, the visualization — taught me:

how to diagnose model failures

how to design robust ML systems

how to combine engineering and research thinking

how to work iteratively and scientifically

This project shaped my interest in robustness, reliable ML, and audio intelligence.
It reflects both curiosity and a commitment to building systems that work in real-world settings.
