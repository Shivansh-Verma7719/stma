-- Table Definition
CREATE TABLE "public"."stock_prices" (
    "id" int4 NOT NULL,
    "ticker" text NOT NULL,
    "date" timestamp NOT NULL,
    "open" numeric NOT NULL,
    "high" numeric NOT NULL,
    "low" numeric NOT NULL,
    "close" numeric NOT NULL,
    "volume" int8 NOT NULL,
    "created_at" timestamp DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "fk_stock_prices_ticker" FOREIGN KEY ("ticker") REFERENCES "public"."companies"("symbol"),
    PRIMARY KEY ("id")
);