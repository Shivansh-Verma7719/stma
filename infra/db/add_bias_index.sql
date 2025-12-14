-- Sequence for bias_index
CREATE SEQUENCE IF NOT EXISTS bias_index_id_seq;

-- Table Definition for bias_index
CREATE TABLE "public"."bias_index" (
    "id" int4 NOT NULL DEFAULT nextval('bias_index_id_seq'::regclass),
    "ticker" text NOT NULL,
    "date" timestamp NOT NULL,
    "open" numeric NOT NULL,
    "high" numeric NOT NULL,
    "low" numeric NOT NULL,
    "close" numeric NOT NULL,
    "volume" int8 NOT NULL,
    "bias_score" numeric,
    "created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "fk_bias_index_ticker" FOREIGN KEY ("ticker") REFERENCES "public"."companies"("symbol"),
    PRIMARY KEY ("id")
);
