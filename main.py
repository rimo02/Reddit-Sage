import asyncpraw
from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_agent = 'Scraper 1.0 by /u/Neither-Trick2134'


reddit = asyncpraw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)


async def search_reddit(query, limit=10):
    subreddit = await reddit.subreddit('all')
    posts = []
    async for submission in subreddit.search(query, limit=limit):
        post = {
            'title': submission.title,
            'id': submission.id,
            'url': submission.url,
            'comments': submission.num_comments,
            'body': submission.selftext
        }
        posts.append(post)
    return posts


async def extract_comments(submission_id):
    submission = await reddit.submission(id=submission_id)
    await submission.comments.replace_more(limit=None)
    comments = []

    async def process_comment(comment, parent_id=None):
        comments.append({
            'comment_body': comment.body,
            'comment_score': comment.score,
        })
        for reply in comment.replies:
            await process_comment(reply, parent_id=comment.id)

    for top_level_comment in submission.comments:
        await process_comment(top_level_comment)

    return comments


async def search_reddit_comments(query: str):
    posts = await search_reddit(query=query, limit=10)
    text = ""
    for post in posts:
        comments = await extract_comments(post['id'])
        for item in comments:
            text += (item['comment_body'])
    return text
