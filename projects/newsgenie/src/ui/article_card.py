import streamlit as st

from src.data.normalized_article import NormalizedArticle

# Thumbnail width (px). Using a fixed width avoids `use_container_width=True`, which
# scales images to the full image-column width — on `layout="wide"` that column is
# still a large share of the viewport, so thumbnails looked oversized.
_IMAGE_THUMB_WIDTH_PX = 220


def render_article_card(article: NormalizedArticle) -> None:
    """Render one NormalizedArticle as a two-column card inside an expander."""
    col_img, col_text = st.columns([1, 5])
    with col_img:
        if article.image_url:
            st.image(
                article.image_url,
                width=_IMAGE_THUMB_WIDTH_PX,
                use_container_width=False,
            )
        else:
            st.caption("No image")
    with col_text:
        st.markdown(f"**[{article.title}]({article.url})**")
        st.write(article.summary)
        st.caption(
            f"{article.source_name} · "
            f"{article.published_at.strftime('%b %d, %H:%M UTC')} · "
            f"`{article.provider}`"
        )
