from __future__ import annotations

from polaris_rag.retrieval.document_preprocessor import preprocess_html


def test_preprocess_html_skips_detached_descendants_after_parent_decompose():
    html = """
    <html>
      <body>
        <div role="navigation">
          <div class="toctree-wrapper compound">
            <a href="/child">Child link</a>
          </div>
        </div>
        <main><p>Keep me</p></main>
      </body>
    </html>
    """

    processed = preprocess_html(
        html=html,
        tags=["script"],
        conditions=[
            {"tag": "div", "condition": "element.get('role', '').strip().lower() == 'navigation'"},
            {"tag": "div", "condition": "'class' in element.attrs and element.attrs['class'] == ['toctree-wrapper', 'compound']"},
            {"tag": "a", "condition": "not element.parent.find_all(string=True, recursive=False)"},
        ],
    )

    assert "Keep me" in processed
    assert "Child link" not in processed
