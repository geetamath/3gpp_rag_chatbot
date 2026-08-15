import { useEffect, useState } from "react";
import axios from "axios";
import {
  Search, ShieldCheck, Database, FileText, ArrowUpRight,
  Activity, Copy, Check, Menu, X, AlertTriangle,
} from "lucide-react";
import "./App.css";

// In production (single-container deploy), the frontend is served by the
// same FastAPI app, so same-origin relative "/api" is correct by default.
// For local dev (frontend on :3000, backend on :8000 separately), set
// REACT_APP_BACKEND_URL=http://localhost:8000 in webapp/frontend/.env.local
const API = `${process.env.REACT_APP_BACKEND_URL || ""}/api`;

const prompts = [
  "What does the AMF do?",
  "Describe the PDU session establishment procedure",
  "What is network slicing?",
];

function SourceItem({ source }) {
  return (
    <div className="source-item" data-testid={`source-${source.id}`}>
      <div className="doc-icon"><FileText size={18} /></div>
      <div>
        <strong>{source.title}</strong>
        <span>{source.subtitle}</span>
        <small>{source.release} · {source.chunks} chunks indexed</small>
      </div>
      <span className="indexed"><span />{source.status}</span>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(null);
  const [mobileNav, setMobileNav] = useState(false);

  useEffect(() => {
    axios.get(`${API}/health`).then(r => setHealth(r.data)).catch(() => setHealth({ status: "unreachable" }));
    axios.get(`${API}/sources`).then(r => setSources(r.data.sources)).catch(() => {});
  }, []);

  const ask = async (value = question) => {
    if (!value.trim() || loading) return;
    setQuestion("");
    setMessages(m => [...m, { role: "user", text: value }]);
    setLoading(true);
    try {
      const r = await axios.post(`${API}/query`, { question: value });
      setMessages(m => [...m, { role: "assistant", ...r.data }]);
    } catch (err) {
      const detail = err?.response?.data?.detail || "The evidence index is temporarily unavailable.";
      setMessages(m => [...m, { role: "assistant", answer: detail, grounded: false, citations: [], status: "ERROR" }]);
    } finally {
      setLoading(false);
    }
  };

  const copy = (text, i) => {
    navigator.clipboard?.writeText(text);
    setCopied(i);
    setTimeout(() => setCopied(null), 1400);
  };

  const indexReady = health && health.status === "ok";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">3G</div>
          <div><b>3GPP<span>RAG</span></b><small>Evidence-grounded telecom intelligence</small></div>
        </div>
        <button className="menu-btn" onClick={() => setMobileNav(!mobileNav)}>
          {mobileNav ? <X /> : <Menu />}
        </button>
        <nav className={mobileNav ? "nav open" : "nav"}>
          <a className="active" href="#chat">Ask the standards</a>
          <a href="#sources">Source library <ArrowUpRight size={14} /></a>
          <div className={`system-status ${indexReady ? "" : "status-warn"}`}>
            <span />{indexReady ? "INDEX ONLINE" : health ? "INDEX NOT BUILT" : "CONNECTING…"}
          </div>
        </nav>
      </header>

      <main className="layout">
        <section className="workspace" id="chat">
          <div className="eyebrow"><ShieldCheck size={15} /> EVIDENCE-GATED + VERIFIED <span className="line" /></div>
          <h1>Ask the<br /><em>standards.</em></h1>
          <p className="intro">
            A retrieval-augmented assistant for 3GPP specifications.<br />
            Every answer is gated on retrieval confidence and verified against its cited clause before being shown.
          </p>

          {!indexReady && health && (
            <div className="index-warning">
              <AlertTriangle size={16} />
              <span>
                Index not built yet on the backend. Run <code>python -m src.ingest</code> and
                <code> python -m src.build_index</code>, then reload this page.
              </span>
            </div>
          )}

          <div className="chat-area">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="signal"><span /><span /><span /><span /></div>
                <strong>What would you like to verify?</strong>
                <small>Ask a question in plain language. The assistant will only answer when retrieval evidence clears a confidence threshold — otherwise it refuses.</small>
              </div>
            )}
            {messages.map((m, i) => m.role === "user" ? (
              <div className="message user-message" key={i}>{m.text}</div>
            ) : (
              <div className="answer-block" key={i}>
                <div className="answer-head">
                  <span className={m.grounded ? "verified" : "unverified"}>
                    {m.grounded ? <><ShieldCheck size={15} /> VERIFIED AGAINST SOURCE</> : <>{m.status === "ERROR" ? "ERROR" : "NO VERIFIED EVIDENCE"}</>}
                  </span>
                  {m.grounded && <span className="confidence">{m.confidence}% claims verified</span>}
                </div>
                <p>{m.answer}</p>
                {m.citations?.length > 0 && (
                  <div className="citations">
                    <label>Retrieved evidence</label>
                    {m.citations.map(c => (
                      <div className="citation" key={c.id}>
                        <span>{c.document}</span>
                        <b>§ {c.section}</b>
                        <small>{c.release}{c.page ? `, p.${c.page}` : ""}</small>
                      </div>
                    ))}
                  </div>
                )}
                {m.answer && (
                  <button className="copy-btn" onClick={() => copy(m.answer, i)}>
                    {copied === i ? <Check size={14} /> : <Copy size={14} />} {copied === i ? "Copied" : "Copy answer"}
                  </button>
                )}
              </div>
            ))}
            {loading && <div className="answer-block loading-block"><span className="dot-flash" />Retrieving and verifying evidence…</div>}
          </div>

          <div className="prompt-row">
            {prompts.map(p => <button key={p} onClick={() => ask(p)}>{p}</button>)}
          </div>
          <form className="ask-form" onSubmit={e => { e.preventDefault(); ask(); }}>
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ask about a 3GPP requirement, procedure, or interface…"
            />
            <button disabled={loading || !question.trim()}>
              <Search size={18} /><span>{loading ? "Retrieving…" : "Ask"}</span>
            </button>
          </form>
          <div className="form-note">
            <span><ShieldCheck size={13} /> Answers are gated on retrieval confidence and post-generation verified</span>
            <span>Press Enter to submit</span>
          </div>
        </section>

        <aside className="sidebar">
          <div className="side-card telemetry">
            <div className="side-label"><Activity size={14} /> RETRIEVAL TELEMETRY</div>
            <div className="metric"><strong>{health?.documents ?? "–"}</strong><span>Documents indexed</span></div>
            <div className="metric"><strong>{health?.chunks ?? "–"}</strong><span>Evidence chunks</span></div>
            <div className="metric"><strong>{health?.llm_backend ?? "–"}</strong><span>LLM backend</span></div>
          </div>

          <div className="side-card library" id="sources">
            <div className="side-label"><Database size={14} /> SOURCE LIBRARY <span>{sources.length}</span></div>
            {sources.length === 0 && <p className="empty-sources">No documents indexed yet — add files to <code>data/raw/</code> and rebuild the index.</p>}
            {sources.map(s => <SourceItem source={s} key={s.id} />)}
          </div>

          <div className="guardrail">
            <ShieldCheck size={20} />
            <div>
              <b>Evidence gate + claim verification</b>
              <span>Low-confidence retrievals are refused before generation. Every generated claim is checked against its cited chunk before being shown.</span>
            </div>
          </div>
        </aside>
      </main>

      <footer>
        <span>3GPP RAG WORKBENCH</span>
        <span>Embedding backend <b>{health?.embedding_backend ?? "–"}</b></span>
        <span>System status <i>{indexReady ? "● operational" : "○ index not built"}</i></span>
      </footer>
    </div>
  );
}
