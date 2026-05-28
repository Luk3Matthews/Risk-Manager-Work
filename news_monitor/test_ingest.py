"""Quick test: fetch a few articles, tag them, store in DB, then launch Streamlit."""
import time
from src import db, news_gdelt
from src.tagger import tag_article
from src.taxonomy import get_taxonomy

db.init_db()
tax = get_taxonomy()

queries = ["OPEC production cut", "Fed rate hike", "China growth slowdown"]
total = 0

for q in queries:
    print(f"Fetching: {q}...")
    try:
        articles = news_gdelt.fetch_articles(q, max_records=5)
    except Exception as e:
        print(f"  ERROR: {e}")
        articles = []
    print(f"  Got {len(articles)} articles")
    for a in articles:
        aid = db.insert_article(
            source="gdelt",
            title=a["title"],
            snippet=a["snippet"],
            url=a["url"],
            published_at=a["published_at"],
            query=q,
        )
        if aid:
            tags = tag_article(
                title=a["title"],
                snippet=a["snippet"],
                source_reliability=0.6,
                taxonomy_instance=tax,
            )
            if tags:
                db.insert_tags(aid, tags)
                total += 1
                print(f"    Tagged: {a['title'][:55]}... ({len(tags)} tags)")
            else:
                print(f"    No tags: {a['title'][:55]}")
                total += 1  # still count stored articles
    time.sleep(8)  # respect GDELT rate limit

print(f"\nTotal new articles ingested: {total}")
recent = db.get_recent_articles(hours=1)
print(f"Articles in DB (last hour): {len(recent)}")
heatmap = db.get_factor_heatmap(hours=1)
print(f"Active factors: {list(heatmap.keys())}")
for f, d in heatmap.items():
    print(f"  {f}: {d['count']} signals, conf={d['avg_confidence']:.1%}")
