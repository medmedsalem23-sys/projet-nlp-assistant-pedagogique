"""Interface Streamlit : Résumé / Quiz / Chat."""
import streamlit as st
from src.rag import extract_text, chunk_text, Index
from src.features import summarize, generate_quiz, chat_answer

REFUS = "Je ne trouve pas la réponse dans le document."

st.set_page_config(page_title="EduAssist · Assistant pédagogique", page_icon="🎓", layout="wide")

st.markdown("""<style>
.block-container{max-width:1050px;padding-top:2rem}
.hero{background:linear-gradient(135deg,#1e3a8a,#4f46e5);color:#fff;padding:1.5rem 2rem;
      border-radius:16px;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:1.8rem;color:#fff}
.hero p{margin:.3rem 0 0;opacity:.9}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;
       background:#eef2ff;color:#3730a3;font-weight:600}
.stTabs [data-baseweb="tab"]{font-weight:600}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🎓 EduAssist</h1>
<p>Résumés, quiz et questions-réponses sur vos cours, avec citation des sources.</p></div>""",
            unsafe_allow_html=True)

# ---------- Barre latérale ----------
with st.sidebar:
    st.header("📄 Document")
    pdf = st.file_uploader("Téléversez un cours (PDF)", type="pdf")
    with st.expander("⚙️ Paramètres avancés"):
        size = st.slider("Taille des chunks", 500, 2000, 1000, 100)
        overlap = st.slider("Chevauchement", 0, 400, 150, 50)
        k = st.slider("Passages récupérés (k)", 1, 8, 4)
        threshold = st.slider("Seuil de similarité (0 = désactivé)", 0.0, 0.8, 0.0, 0.05,
                              help="Si le meilleur passage est en dessous du seuil, le chatbot refuse sans appeler le LLM.")

if not pdf:
    st.info("👈 Téléversez un cours en PDF dans la barre latérale pour commencer.")
    st.stop()

key = (pdf.name, size, overlap)
if st.session_state.get("key") != key:
    with st.spinner("Indexation du document..."):
        chunks = chunk_text(extract_text(pdf), size, overlap)
        st.session_state.update(key=key, chunks=chunks, index=Index(chunks),
                                history=[], quiz=[], summaries={})
chunks, index = st.session_state.chunks, st.session_state.index

c1, c2, c3 = st.columns(3)
c1.metric("Document", pdf.name[:22])
c2.metric("Chunks indexés", len(chunks))
c3.metric("Caractères", f"{sum(len(c) for c in chunks):,}".replace(",", " "))

tab1, tab2, tab3 = st.tabs(["📝 Résumé", "🧩 Quiz", "💬 Chat"])

# ---------- Résumé ----------
with tab1:
    level = st.radio("Niveau de détail", ["court", "moyen", "détaillé"], horizontal=True)
    if st.button("Générer le résumé", type="primary"):
        try:
            with st.spinner("Résumé en cours..."):
                st.session_state.summaries[level] = summarize(chunks, level)
        except Exception as e:
            st.error(f"Erreur lors de l'appel au LLM : {e}")
    if level in st.session_state.summaries:
        text = st.session_state.summaries[level]
        with st.container(border=True):
            st.markdown(text)
        st.download_button("⬇️ Télécharger le résumé", text, file_name=f"resume_{level}.txt")

# ---------- Quiz ----------
with tab2:
    a, b = st.columns(2)
    n = a.slider("Nombre de questions", 3, 10, 5)
    diff = b.selectbox("Difficulté", ["facile", "moyen", "difficile"], index=1)
    if st.button("Générer le quiz", type="primary"):
        try:
            with st.spinner("Génération du quiz..."):
                st.session_state.quiz = generate_quiz(chunks, n, diff)
        except Exception:
            st.error("Le modèle n'a pas renvoyé un quiz valide. Réessayez.")
    quiz = st.session_state.quiz
    score = answered = 0
    for i, q in enumerate(quiz):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {q['question']}**")
            choice = st.radio("Réponse", q["choices"], index=None, key=f"q{i}_{len(quiz)}",
                              label_visibility="collapsed")
            if choice is not None:
                answered += 1
                good = q["choices"][q["answer"]]
                if choice == good:
                    score += 1
                    st.success(f"Correct. {q['explanation']}")
                else:
                    st.error(f"Faux. Bonne réponse : {good}. {q['explanation']}")
    if quiz:
        st.metric("Score", f"{score} / {len(quiz)}", help=f"{answered} question(s) répondue(s)")

# ---------- Chat ----------
with tab3:
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.history = []
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.write(m["text"])
            if m.get("hits"):
                with st.expander("📚 Passages utilisés"):
                    for n_, (_, s, t) in enumerate(m["hits"]):
                        st.markdown(f"<span class='badge'>[{n_ + 1}] similarité {s:.2f}</span>",
                                    unsafe_allow_html=True)
                        st.caption(t)
    if not st.session_state.history:
        st.caption("Exemples de questions :")
        cols = st.columns(2)
        examples = ["Fais un résumé des points essentiels du cours", "Quels sont les concepts clés ?"]
        for col, ex in zip(cols, examples):
            if col.button(ex):
                st.session_state.pending = ex
    question = st.chat_input("Posez une question sur le cours") or st.session_state.pop("pending", None)
    if question:
        st.session_state.history.append({"role": "user", "text": question})
        try:
            hits = index.search(question, k)
            if threshold > 0 and (not hits or hits[0][1] < threshold):
                answer = REFUS
            else:
                answer, hits = chat_answer(index, question, k)
        except Exception as e:
            answer, hits = f"Erreur : {e}", []
        st.session_state.history.append({"role": "assistant", "text": answer, "hits": hits})
        st.rerun()
