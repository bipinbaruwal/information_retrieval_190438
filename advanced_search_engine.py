import threading
import webbrowser
import re
import csv
import os
import sys
import joblib
import darkdetect
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

# --- UI / Theming ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
except Exception as e:
    print("ttkbootstrap is required for the advanced UI. Install it via 'pip install ttkbootstrap'.")
    raise

# -------------------------
# CONFIG — matches reference
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_FILE = os.path.join(DATA_DIR, 'index.pkl')


# -------------------------
# DATA STRUCTURES
# -------------------------
@dataclass
class Publication:
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    link: str
    relevancy_score: float


# -------------------------
# RANKER 
# -------------------------
from sklearn.metrics.pairwise import cosine_similarity

class Ranker:
    def __init__(self, vectorizer, tfidf_matrix):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

    def rank(self, query: str) -> List[Tuple[int, float]]:
        if not query:
            return []
        query_vector = self.vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        doc_scores = [(doc_id, score) for doc_id, score in enumerate(cosine_scores) if score > 0]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores


# -------------------------
# ENGINE 
# -------------------------
class VerticalSearchEngine:
    """
    Loads the same index layout your reference uses:
      {
        'positional_index': ...,
        'doc_store': list[dict],
        'vectorizer': ...,
        'tfidf_matrix': ...
      }
    and exposes search(query, top_k) identical to the reference.
    """
    def __init__(self):
        print("Initializing search engine...")
        try:
            # Ensure data directory exists
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
                raise FileNotFoundError(f"Created data directory at {DATA_DIR}. Please run indexer first.")
                
            if not os.path.exists(INDEX_FILE):
                raise FileNotFoundError(
                    f"Index file not found at {INDEX_FILE}\n"
                    "Please run 'python -m IR_Bipin.indexer.build_index' first."
                )
                
            print(f"Loading index from {INDEX_FILE}...")
            index_data = joblib.load(INDEX_FILE)
            
            # Validate index structure
            required_keys = ['positional_index', 'doc_store', 'vectorizer', 'tfidf_matrix']
            missing_keys = [key for key in required_keys if key not in index_data]
            
            if missing_keys:
                raise ValueError(
                    f"Invalid index structure. Missing keys: {', '.join(missing_keys)}\n"
                    "Please rebuild the index using the latest indexer version."
                )
                
            self.positional_index = index_data['positional_index']
            self.doc_store = index_data['doc_store']
            
            if not self.doc_store:
                raise ValueError("Document store is empty. Please rebuild index with documents.")
                
            self.ranker = Ranker(index_data['vectorizer'], index_data['tfidf_matrix'])
            print(f"Successfully loaded {len(self.doc_store)} documents.")
            
        except Exception as e:
            print(f"Error initializing search engine: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 1000) -> List[Publication]:
        if not query.strip():
            return []
            
        try:
            scores = self.ranker.rank(query)
            results = []
            for doc_id, score in scores[:top_k]:
                doc_info = self.doc_store[doc_id]
                results.append(
                    Publication(
                        title=doc_info.get("title", "No Title"),
                        authors=list(doc_info.get("authors", [])),
                        abstract=doc_info.get("abstract", ""),
                        published_date=doc_info.get("published_date", "N/A"),
                        link=doc_info.get("link", ""),
                        relevancy_score=round(float(score), 4),
                    )
                )
            return results
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

# -------------------------
# UTILITIES
# -------------------------
def split_authors(authors: List[str]) -> str:
    return ", ".join(a for a in authors if a)

def highlight_text(widget: tk.Text, pattern: str, tag: str, start="1.0", end="end"):
    widget.tag_remove(tag, start, end)
    if not pattern.strip():
        return
    try:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    except re.error:
        return
    idx = start
    while True:
        idx = widget.search(regex, idx, nocase=True, stopindex=end, regexp=True)
        if not idx:
            break
        lastidx = f"{idx}+{len(widget.get(idx, idx + ' lineend'))}c"
        # find actual match span
        line_text = widget.get(idx, f"{idx} lineend")
        for match in regex.finditer(line_text):
            s = f"{idx}+{match.start()}c"
            e = f"{idx}+{match.end()}c"
            widget.tag_add(tag, s, e)
        idx = f"{idx} lineend + 1c"


def parse_date_safe(s: str):
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


# -------------------------
# ADVANCED UI
# -------------------------
class SearchApp:
    PAGE_SIZE_DEFAULT = 10

    def __init__(self, engine: VerticalSearchEngine):
        self.engine = engine
        self.results: List[Publication] = []
        self.filtered: List[Publication] = []

        # Theme bootstrap
        dark = darkdetect.isDark()
        theme = "darkly" if dark else "flatly"
        self.root = tb.Window(title="Research Search — Advanced", themename=theme)
        self.root.geometry("1280x840")
        self.root.minsize(1024, 720)

        # State
        self.query_var = tk.StringVar()
        self.page_var = tk.IntVar(value=1)
        self.page_size_var = tk.IntVar(value=SearchApp.PAGE_SIZE_DEFAULT)

        # Build UI
        self._build_layout()
        self._bind_keys()

    def _build_layout(self):
        # Top bar
        top = ttk.Frame(self.root, padding=12)
        top.pack(side=tk.TOP, fill=tk.X)

        self.search_entry = ttk.Entry(top, textvariable=self.query_var, width=80)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.search_entry.focus_set()

        self.search_btn = ttk.Button(top, text="Search", bootstyle=PRIMARY, command=self._on_search_clicked)
        self.search_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.clear_btn = ttk.Button(top, text="Clear", command=self._on_clear)
        self.clear_btn.pack(side=tk.LEFT)

        ttk.Label(top, text=" Page size: ").pack(side=tk.LEFT, padx=(16, 4))
        self.page_size_spin = ttk.Spinbox(top, from_=5, to=100, textvariable=self.page_size_var, width=5)
        self.page_size_spin.pack(side=tk.LEFT)

        # Main split — results list on left, details on right
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left pane
        left = ttk.Frame(main)
        main.add(left, weight=3)

        # Header / summary
        self.summary_var = tk.StringVar(value="Type a query and press Enter.")
        self.summary_label = ttk.Label(left, textvariable=self.summary_var)
        self.summary_label.pack(side=tk.TOP, anchor="w", pady=(0, 6))

        # Results Text (card-like)
        self.text = tk.Text(left, wrap="word", relief="flat", bd=0, padx=10, pady=10)
        self.text.pack(fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.place(in_=self.text, relx=1.0, rely=0, relheight=1.0, anchor="ne")

        # Text tags for styling
        self.text.tag_configure("h1", font=("TkDefaultFont", 12, "bold"))
        self.text.tag_configure("muted", foreground="#8a8a8a")
        self.text.tag_configure("link", foreground="#1f6feb", underline=1)
        self.text.tag_configure("chip", background="#e9ecef")
        self.text.tag_configure("score", foreground="#0d6efd")
        self.text.tag_configure("hl", background="#ffd54d")

        self.text.bind("<Button-1>", self._maybe_open_link)

        # Pagination bar
        pager = ttk.Frame(left)
        pager.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))
        self.prev_btn = ttk.Button(pager, text="◀ Prev", command=self._prev_page)
        self.next_btn = ttk.Button(pager, text="Next ▶", command=self._next_page)
        self.page_label_var = tk.StringVar(value="")
        self.page_label = ttk.Label(pager, textvariable=self.page_label_var)
        self.prev_btn.pack(side=tk.LEFT)
        self.page_label.pack(side=tk.LEFT, padx=8)
        self.next_btn.pack(side=tk.LEFT)

        # Right pane — details
        right = ttk.Frame(main, padding=(8, 0, 0, 0))
        main.add(right, weight=2)

        ttk.Label(right, text="Details", style="secondary.TLabel").pack(anchor="w")
        self.detail = tk.Text(right, wrap="word", height=10, relief="solid", bd=1, padx=10, pady=10)
        self.detail.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        detail_btns = ttk.Frame(right)
        detail_btns.pack(fill=tk.X)
        self.open_btn = ttk.Button(detail_btns, text="Open Link", command=self._open_selected_link)
        self.copy_link_btn = ttk.Button(detail_btns, text="Copy Link", command=self._copy_selected_link)
        self.copy_cite_btn = ttk.Button(detail_btns, text="Copy Citation", command=self._copy_selected_citation)
        self.export_btn = ttk.Button(detail_btns, text="Export Page CSV", command=self._export_current_page)
        self.open_btn.pack(side=tk.LEFT)
        self.copy_link_btn.pack(side=tk.LEFT, padx=6)
        self.copy_cite_btn.pack(side=tk.LEFT)
        self.export_btn.pack(side=tk.RIGHT)

    def _bind_keys(self):
        self.root.bind("<Return>", lambda e: self._on_search_clicked())
        self.root.bind("<Control-l>", lambda e: self._focus_search())
        self.root.bind("<Control-L>", lambda e: self._focus_search())
        self.root.bind("<Control-e>", lambda e: self._export_current_page())
        self.root.bind("<Control-E>", lambda e: self._export_current_page())
        self.root.bind("<Control-c>", lambda e: self._copy_selected_link())
        self.root.bind("<Control-C>", lambda e: self._copy_selected_link())

    # ------------- Actions -------------
    def _focus_search(self):
        self.search_entry.focus_set()
        self.search_entry.selection_range(0, tk.END)

    def _on_clear(self):
        self.query_var.set("")
        self.text.delete("1.0", tk.END)
        self.detail.delete("1.0", tk.END)
        self.summary_var.set("Type a query and press Enter.")
        self.page_var.set(1)
        self.results.clear()
        self.filtered.clear()
        self._update_pager()

    def _on_search_clicked(self):
        q = self.query_var.get().strip()
        if not q:
            return
        self._set_busy(True)
        self.text.delete("1.0", tk.END)
        self.detail.delete("1.0", tk.END)
        self.summary_var.set("Searching…")

        def worker():
            try:
                results = self.engine.search(q, top_k=1000)  # keep logic identical
            except Exception as e:
                self.root.after(0, lambda: self._on_search_error(e))
                return
            self.root.after(0, lambda: self._on_search_done(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_search_error(self, e: Exception):
        self._set_busy(False)
        messagebox.showerror("Search Error", str(e))

    def _on_search_done(self, results: List[Publication]):
        self._set_busy(False)
        self.results = results
        self.filtered = results  # no extra filtering by default (keeps logic intact)
        total = len(self.filtered)
        q = self.query_var.get().strip()
        self.summary_var.set(f"Found {total} result(s) for “{q}”.")
        self.page_var.set(1)
        self._render_page()

    def _set_busy(self, busy: bool):
        try:
            self.root.configure(cursor="watch" if busy else "")
            self.search_btn.configure(state=("disabled" if busy else "normal"))
        except Exception:
            pass

    # ------------- Pagination & Rendering -------------
    def _get_page_slice(self):
        page = max(1, self.page_var.get())
        page_size = max(1, self.page_size_var.get())
        start = (page - 1) * page_size
        end = start + page_size
        return start, end

    def _render_page(self):
        self.text.delete("1.0", tk.END)
        start, end = self._get_page_slice()
        q = self.query_var.get().strip()
        page_items = self.filtered[start:end]
        if not page_items:
            self.text.insert(tk.END, "No results to display on this page.")
            self._update_pager()
            return

        for idx, res in enumerate(page_items, start=start + 1):
            # Title
            self.text.insert(tk.END, f"{idx}. ", ("muted",))
            title_start = self.text.index(tk.END)
            self.text.insert(tk.END, res.title + "\n", ("h1",))
            title_end = self.text.index(tk.END)

            # Chips: date + score
            self.text.insert(tk.END, "Published: ", ("muted",))
            self.text.insert(tk.END, f"{res.published_date}  ")
            self.text.insert(tk.END, "• Relevancy: ", ("muted",))
            self.text.insert(tk.END, f"{res.relevancy_score}\n", ("score",))

            # Authors
            if res.authors:
                self.text.insert(tk.END, "Authors: ", ("muted",))
                self.text.insert(tk.END, split_authors(res.authors) + "\n")

            # Abstract
            if res.abstract:
                self.text.insert(tk.END, "Abstract: ", ("muted",))
                abs_start = self.text.index(tk.END)
                self.text.insert(tk.END, res.abstract.strip() + "\n")
                abs_end = self.text.index(tk.END)
            else:
                abs_start = abs_end = self.text.index(tk.END)

            # Link
            self.text.insert(tk.END, "URL: ", ("muted",))
            link_start = self.text.index(tk.END)
            self.text.insert(tk.END, res.link or "N/A", ("link",))
            link_end = self.text.index(tk.END)
            self.text.insert(tk.END, "\n\n")

            # Map clickable range to this result index
            self.text.tag_add(f"link-{idx}", link_start, link_end)
            self.text.tag_bind(f"link-{idx}", "<Button-1>", lambda e, url=res.link: self._open_url(url))

            # Live highlight of query
            highlight_text(self.text, q, "hl", start=title_start, end=title_end)
            highlight_text(self.text, q, "hl", start=abs_start, end=abs_end)

        self._update_pager()

        # Select first item to show details
        if page_items:
            self._show_details(page_items[0])

        # Scroll to top of page
        self.text.see("1.0")

    def _update_pager(self):
        total = len(self.filtered)
        page_size = max(1, self.page_size_var.get())
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, self.page_var.get()), total_pages)
        self.page_var.set(page)
        self.prev_btn.configure(state=("disabled" if page <= 1 else "normal"))
        self.next_btn.configure(state=("disabled" if page >= total_pages else "normal"))
        self.page_label_var.set(f"Page {page} / {total_pages}")

    def _prev_page(self):
        self.page_var.set(max(1, self.page_var.get() - 1))
        self._render_page()

    def _next_page(self):
        self.page_var.set(self.page_var.get() + 1)
        self._render_page()

    # ------------- Selection & Details -------------
    def _current_page_selection(self):
        start, end = self._get_page_slice()
        try:
            # Heuristic: pick the first URL on screen to derive selection
            idxRanges = self.text.tag_ranges("link")
            if idxRanges:
                return self.filtered[start:end][0] if self.filtered[start:end] else None
        except Exception:
            pass
        return self.filtered[start:end][0] if self.filtered[start:end] else None

    def _show_details(self, res: Publication):
        self.detail.delete("1.0", tk.END)
        self.detail.insert(tk.END, f"Title: {res.title}\n\n")
        self.detail.insert(tk.END, f"Published: {res.published_date}\n")
        self.detail.insert(tk.END, f"Relevancy: {res.relevancy_score}\n\n")
        if res.authors:
            self.detail.insert(tk.END, "Authors:\n")
            for a in res.authors:
                self.detail.insert(tk.END, f"  • {a}\n")
            self.detail.insert(tk.END, "\n")
        if res.abstract:
            self.detail.insert(tk.END, "Abstract:\n")
            self.detail.insert(tk.END, res.abstract.strip() + "\n\n")
        self.detail.insert(tk.END, f"URL:\n{res.link or 'N/A'}")

    # ------------- Link handling -------------
    def _maybe_open_link(self, event):
        # If clicked somewhere, try to detect a tagged link
        index = self.text.index(f"@{event.x},{event.y}")
        for tag in self.text.tag_names(index):
            if tag.startswith("link-"):
                # actual binding on tag already opens URL; we also try to update details
                start, end = self._get_page_slice()
                if self.filtered[start:end]:
                    self._show_details(self.filtered[start:end][0])
                break

    def _open_url(self, url: str):
        if url:
            webbrowser.open_new_tab(url)

    def _open_selected_link(self):
        res = self._current_page_selection()
        if res and res.link:
            webbrowser.open_new_tab(res.link)

    def _copy_selected_link(self):
        res = self._current_page_selection()
        if not res or not res.link:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(res.link)
        self.root.update()
        messagebox.showinfo("Copied", "Link copied to clipboard.")

    def _copy_selected_citation(self):
        res = self._current_page_selection()
        if not res:
            return
        authors = split_authors(res.authors)
        cite = f"{authors} ({res.published_date}). {res.title}. {res.link}"
        self.root.clipboard_clear()
        self.root.clipboard_append(cite)
        self.root.update()
        messagebox.showinfo("Copied", "Citation copied to clipboard.")

    # ------------- Export -------------
    def _export_current_page(self):
        start, end = self._get_page_slice()
        rows = self.filtered[start:end]
        if not rows:
            messagebox.showinfo("Export", "Nothing to export on this page.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="search_page.csv",
            title="Export current page to CSV"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Authors", "Published", "Score", "Link", "Abstract"])
            for r in rows:
                writer.writerow([r.title, "; ".join(r.authors), r.published_date, r.relevancy_score, r.link, r.abstract])
        messagebox.showinfo("Export", f"Saved: {os.path.basename(path)}")

    # ------------- Run -------------
    def run(self):
        # Center the window
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.mainloop()


def main():
    try:
        print(f"Using data directory: {DATA_DIR}")
        engine = VerticalSearchEngine()
        app = SearchApp(engine)
        app.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        messagebox.showerror("Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
