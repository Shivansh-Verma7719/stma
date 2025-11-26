-- Schema for Media Outlets
CREATE TABLE IF NOT EXISTS media_outlets (
    id SERIAL PRIMARY KEY,
    domain TEXT UNIQUE NOT NULL,
    name TEXT,
    social_handles JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Schema for Companies
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    symbol TEXT UNIQUE NOT NULL,
    current_page INTEGER DEFAULT 0,
    is_processed BOOLEAN DEFAULT FALSE,
    last_error TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
    company_id INTEGER REFERENCES companies(id),
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_company_id ON articles(company_id);
CREATE INDEX IF NOT EXISTS idx_articles_media_outlet_id ON articles(media_outlet_id);
