-- Schema for Media Outlets
CREATE TABLE IF NOT EXISTS media_outlets (
    id SERIAL PRIMARY KEY,
    domain TEXT UNIQUE NOT NULL,
    name TEXT,
    social_handles JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema for Articles
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT UNIQUE NOT NULL,
    source TEXT,
    published_at TIMESTAMP,
    media_outlet_id INTEGER REFERENCES media_outlets(id),
    social_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema for Reddit Posts
CREATE TABLE IF NOT EXISTS reddit_posts (
    id SERIAL PRIMARY KEY,
    reddit_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    url TEXT,
    subreddit TEXT,
    author TEXT,
    score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
