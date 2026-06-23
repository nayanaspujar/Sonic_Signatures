
import streamlit as st
import pickle
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import Counter

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sonic Signatures",
    layout="wide"
)

st.title("🎵 Sonic Signatures")
st.markdown(
    "### Audio Fingerprinting and Song Recognition System"
)

# =====================================================
# LOAD DATABASE
# =====================================================

@st.cache_resource
def load_database():

    with open("fingerprint_database.pkl", "rb") as f:
        db = pickle.load(f)

    return db

database = load_database()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Database Information")

st.sidebar.metric(
    "Songs Indexed",
    len(database)
)

st.sidebar.metric(
    "Fingerprint Type",
    "(f1,f2,Δt)"
)

with st.sidebar.expander("View Songs"):

    for song in sorted(database.keys()):
        st.write(song)

# =====================================================
# FINGERPRINT GENERATION
# =====================================================

def generate_fingerprints(audio_file):

    audio_file.seek(0)

    y, sr = librosa.load(
        audio_file,
        sr=8000,
        duration=60
    )

    f, t, Sxx = spectrogram(
        y,
        fs=sr,
        nperseg=1024,
        noverlap=512
    )

    Sxx_db = 10 * np.log10(
        Sxx + 1e-10
    )

    local_max = maximum_filter(
        Sxx_db,
        size=20
    )

    peaks = np.where(
        (Sxx_db == local_max)
        &
        (
            Sxx_db >
            np.percentile(
                Sxx_db,
                99
            )
        )
    )

    fingerprints = []

    for i in range(
        len(peaks[0]) - 5
    ):

        f1 = peaks[0][i]
        t1 = peaks[1][i]

        for j in range(1, 6):

            f2 = peaks[0][i + j]
            t2 = peaks[1][i + j]

            dt = t2 - t1

            fingerprints.append(
                (
                    int(f1),
                    int(f2),
                    int(dt)
                )
            )

    return (
        fingerprints,
        f,
        t,
        Sxx_db,
        peaks
    )

# =====================================================
# SONG IDENTIFICATION
# =====================================================

def identify_song(uploaded_file):

    (
        query_fp,
        f,
        t,
        Sxx_db,
        peaks
    ) = generate_fingerprints(
        uploaded_file
    )

    votes = Counter()

    for song, db_fp in database.items():

        matches = len(
            set(query_fp).intersection(
                set(db_fp)
            )
        )

        votes[song] = matches

    best_song = max(
        votes,
        key=votes.get
    )

    return (
        best_song,
        votes,
        f,
        t,
        Sxx_db,
        peaks,
        query_fp
    )

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "🎵 Single Query",
        "📂 Batch Mode"
    ]
)

# =====================================================
# SINGLE QUERY MODE
# =====================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload MP3 or WAV",
        type=[
            "mp3",
            "wav"
        ]
    )

    if uploaded_file:

        st.audio(uploaded_file)

        (
            best_song,
            votes,
            f,
            t,
            Sxx_db,
            peaks,
            query_fp
        ) = identify_song(
            uploaded_file
        )

        total_votes = sum(
            votes.values()
        )

        if total_votes > 0:

            confidence = (
                votes[best_song]
                /
                total_votes
                * 100
            )

        else:

            confidence = 0

        st.success(
            f"""
### 🎵 Match Found

Song: {best_song}

Confidence: {confidence:.2f}%
"""
        )

        st.progress(
            min(
                confidence / 100,
                1.0
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Peaks",
                len(peaks[0])
            )

        with col2:

            st.metric(
                "Fingerprints",
                len(query_fp)
            )

        st.divider()

        # =========================================
        # Spectrogram
        # =========================================

        st.subheader(
            "Spectrogram"
        )

        fig, ax = plt.subplots(
            figsize=(12,4)
        )

        img = ax.pcolormesh(
            t,
            f,
            Sxx_db,
            shading="gouraud"
        )

        plt.colorbar(
            img,
            ax=ax
        )

        ax.set_xlabel(
            "Time (s)"
        )

        ax.set_ylabel(
            "Frequency (Hz)"
        )

        st.pyplot(fig)

        # =========================================
        # Constellation Map
        # =========================================

        st.subheader(
            "Constellation Map"
        )

        fig2, ax2 = plt.subplots(
            figsize=(12,4)
        )

        ax2.scatter(
            t[peaks[1]],
            f[peaks[0]],
            s=8
        )

        ax2.set_xlabel(
            "Time (s)"
        )

        ax2.set_ylabel(
            "Frequency (Hz)"
        )

        ax2.set_title(
            "Detected Peaks"
        )

        st.pyplot(fig2)

        # =========================================
        # Top Matches
        # =========================================

        st.subheader(
            "Top Candidate Songs"
        )

        top = sorted(
            votes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        df = pd.DataFrame(
            top,
            columns=[
                "Song",
                "Matches"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        st.bar_chart(
            df.set_index(
                "Song"
            )
        )

# =====================================================
# BATCH MODE
# =====================================================

with tab2:

    files = st.file_uploader(
        "Upload Multiple Files",
        type=[
            "wav",
            "mp3"
        ],
        accept_multiple_files=True
    )

    if files:

        results = []

        progress = st.progress(0)

        for idx, file in enumerate(files):

            (
                best_song,
                _,
                _,
                _,
                _,
                _,
                _
            ) = identify_song(
                file
            )

            results.append(
                {
                    "filename":
                        file.name,

                    "prediction":
                        best_song.replace(
                            ".mp3",
                            ""
                        )
                }
            )

            progress.progress(
                (idx + 1)
                /
                len(files)
            )

        results_df = pd.DataFrame(
            results
        )

        st.subheader(
            "Batch Results"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(
            "Download results.csv",
            csv,
            "results.csv",
            "text/csv"
        )

        st.success(
            "results.csv generated successfully!"
        )

st.divider()

st.caption(
    "EE200 • Sonic Signatures"
)
